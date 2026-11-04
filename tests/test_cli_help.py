"""``lift --help`` and every command's ``--help`` on a machine without the GPU stack.

lift.cli.build_parser imports every lift.commands module to build the parser, so one command module that
imports torch, transformers, peft, trl or an API SDK at import time breaks ``lift --help``, and every other
command, wherever they are not installed (a laptop with ``.[dev]`` only). The check runs in a fresh
interpreter in which importing any of those packages fails, as it does there: in this process other tests
may have imported them already, which would hide a top-level import. Nothing is written, and nothing but
argparse's help runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import lift

# What a command module must not import when it is imported (each command imports its stack in its handler).
HEAVY = ("torch", "transformers", "tokenizers", "safetensors", "peft", "trl", "accelerate", "datasets",
         "bitsandbytes", "safetune", "auditkit", "curatorkit", "fasttext", "huggingface_hub", "anthropic",
         "openai")
# The P03 command modules and the commands the P03 brief gives each. They are written in parallel, so a
# module this checkout does not have yet is checked from the commit that adds it on.
P03_COMMANDS = {"train": ("train", "register-adapter"), "utility": ("utility",),
                "loglik": ("belebele", "heldout-loss"), "gate": ("gate",)}
OK = {"code": 0, "usage": True}

PROBE = r'''
import argparse, contextlib, importlib, io, json, pkgutil, sys

HEAVY = set(sys.argv[1].split(","))
tried = []


class NotInstalled:
    """Makes every import of a HEAVY package fail, as on a machine where it is not installed."""

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HEAVY:
            tried.append(name)
            raise ImportError(f"No module named {name!r} (not installed here)")
        return None


sys.meta_path.insert(0, NotInstalled())

from lift import cli, commands


class Names:
    """A subparsers action that records the names register(sub) adds."""

    def __init__(self, sub):
        self.sub, self.names = sub, []

    def add_parser(self, name, **kwargs):
        self.names.append(name)
        return self.sub.add_parser(name, **kwargs)

    def __getattr__(self, name):
        return getattr(self.sub, name)


def helped(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            cli.main(argv)
            code = None
        except SystemExit as exc:
            code = exc.code
        except Exception as exc:
            code = f"{type(exc).__name__}: {exc}"
    return {"code": code, "usage": "usage:" in out.getvalue()}


report = {"top": helped(["--help"]), "modules": {}, "errors": {}, "help": {}}
for info in sorted(pkgutil.iter_modules(commands.__path__), key=lambda m: m.name):
    if info.name.startswith("_"):
        continue
    names = Names(argparse.ArgumentParser(prog="lift").add_subparsers())
    try:
        importlib.import_module(f"lift.commands.{info.name}").register(names)
    except Exception as exc:
        report["errors"][info.name] = f"{type(exc).__name__}: {exc}"
        continue
    report["modules"][info.name] = names.names
for name in [n for found in report["modules"].values() for n in found]:
    report["help"][name] = helped([name, "--help"])
report["tried"] = tried
report["loaded"] = sorted(m for m in sys.modules if m.partition(".")[0] in HEAVY)
print(json.dumps(report))
'''


def probe(tmp_path: Path) -> dict:
    """PROBE's report, run on this checkout's lift (the one this process imported) in a fresh interpreter."""
    src = str(Path(lift.__file__).resolve().parents[1])
    path = os.pathsep.join([src, *filter(None, [os.environ.get("PYTHONPATH")])])
    env = {**os.environ, "PYTHONPATH": path, "PYTHONIOENCODING": "utf-8"}
    done = subprocess.run([sys.executable, "-c", PROBE, ",".join(HEAVY)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_lift_help_and_every_commands_help_need_no_gpu_stack(tmp_path):
    report = probe(tmp_path)
    assert report["errors"] == {}
    assert report["top"] == OK
    assert report["tried"] == [] and report["loaded"] == []
    names = [name for found in report["modules"].values() for name in found]
    assert len(names) == len(set(names)), report["modules"]          # one module per command
    assert {name: result for name, result in report["help"].items() if result != OK} == {}
    assert {"smoke", "generate", "judge", "run-dev", "restart", "sync", "remote"} <= set(names)
    for module, commands in P03_COMMANDS.items():
        if module in report["modules"]:
            assert set(commands) <= set(names), (module, report["modules"][module])
