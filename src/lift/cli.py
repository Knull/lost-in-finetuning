"""``lift <command>``: one entry point for every stage.

Each module in ``lift.commands`` registers its own subcommands through ``register(subparsers)``,
so parallel work never edits the same file. Command modules import torch and friends inside the
handler, never at module import, so ``lift --help`` works on a machine without a GPU stack.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys

from lift import commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lift", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for info in sorted(pkgutil.iter_modules(commands.__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"lift.commands.{info.name}")
        register = getattr(module, "register", None)
        if register is None:
            raise RuntimeError(f"lift.commands.{info.name} has no register(subparsers)")
        register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
