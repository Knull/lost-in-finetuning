"""The repair settings of the dev sweep and their configs/variants.yaml entries (P05). No torch.

Each setting is one repair method at one strength from configs/repairs.yaml, applied to one adapter, in
cost order (wiseft < resta < cast < cast_always):

    wiseft-l{lambda:.2f}-{adapter}        params {"lambda"}
    resta-a{alpha:.2f}-{adapter}          params {"alpha", "dare": false, "base", "base_revision"}
    cast-m{strength:.1f}-{adapter}        params {"strength", "layer", "threshold", "comparator", "pool",
    castalways-m{strength:.1f}-{adapter}          "behavior_layers"}

Every entry pins the adapter's hash (``artifacts.adapter``, from the adapter's own registry entry, which
must pin one); a CAST entry also pins its artifact (``artifacts.cast``, the sha256 of
outputs/artifacts/cast/{adapter}/cast.safetensors) and copies the fitted layer, threshold and comparator
into its params, read from the artifact's metadata without torch. ReSta's Base model and its revision are
params, so a re-pinned Base changes the cache key. Families for the per-family selection: wiseft, resta
and cast (cast and cast_always together).

The entries are deterministic: the GPU job (``lift sweep``) writes them into its own checkout, which
dies with the VM, and the laptop writes the same ones with ``lift register-repairs`` after ``lift sync
pull``, to commit before anything reads the sweep's outputs there (``lift utility --judge-only``
compares the registry with the answers' check records). Writing is a text append, as lift.train.lora
registers adapters, so the file's comments stay.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lift import config
from lift.schemas import VariantSpec

FAMILIES = ("wiseft", "resta", "cast")
METHODS = ("wiseft", "resta", "cast", "cast_always")
FAMILY = {"wiseft": "wiseft", "resta": "resta", "cast": "cast", "cast_always": "cast"}
PREFIX = {"wiseft": "wiseft-l", "resta": "resta-a", "cast": "cast-m", "cast_always": "castalways-m"}
DIGITS = {"wiseft": 2, "resta": 2, "cast": 1, "cast_always": 1}
PARAM = {"wiseft": "lambda", "resta": "alpha", "cast": "strength", "cast_always": "strength"}
_NAME = re.compile(r"^(?P<prefix>wiseft-l|resta-a|castalways-m|cast-m)(?P<value>\d+\.\d+)"
                   r"-(?P<adapter>adapter-.+)$")
_SHA = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
NOTE = "P05 repair setting (lift register-repairs)"


@dataclass(frozen=True)
class Setting:
    method: str
    value: float

    @property
    def family(self) -> str:
        return FAMILY[self.method]

    @property
    def tag(self) -> str:
        return f"{PREFIX[self.method]}{self.value:.{DIGITS[self.method]}f}"

    def name(self, adapter: str) -> str:
        return f"{self.tag}-{adapter}"


def parse_name(name: str) -> tuple[Setting, str]:
    """(setting, adapter) of a repaired variant's name."""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"{name!r} is not a repair setting name (configs/variants.yaml naming scheme)")
    method = {v: k for k, v in PREFIX.items()}[m.group("prefix")]
    return Setting(method, float(m.group("value"))), m.group("adapter")


def settings(families: Iterable[str] = FAMILIES) -> list[Setting]:
    """The dev sweep's settings of ``families`` (configs/repairs.yaml), in cost order, then strength."""
    rep = config.load("repairs")
    wanted = set(families)
    unknown = wanted - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown repair families {sorted(unknown)}; choose from {list(FAMILIES)}")
    out: list[Setting] = []
    if "wiseft" in wanted:
        out += [Setting("wiseft", float(x)) for x in rep["wiseft"]["lambdas"]]
    if "resta" in wanted:
        out += [Setting("resta", float(x)) for x in rep["resta"]["alphas"]]
    if "cast" in wanted:
        out += [Setting("cast", float(x)) for x in rep["cast"]["strengths"]]
        out.append(Setting("cast_always", float(rep["cast"]["always_on_strength"])))
    per_family = {f: sum(s.family == f for s in out) for f in wanted}
    if any(n > 4 for n in per_family.values()):
        raise ValueError(f"configs/repairs.yaml gives a family more than 4 settings ({per_family}): the "
                         "equal "
                         "dev budget is at most 4 each")
    return out


def read_cast_info(path: Path) -> dict[str, Any]:
    """The ``lift`` metadata of a CAST artifact (lift.repair.cast), read from the safetensors header
    without torch."""
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        (n,) = struct.unpack("<Q", head)
        if n > 100_000_000:
            raise ValueError(f"{path} is not a safetensors file")
        header = json.loads(f.read(n).decode("utf-8"))
    meta = header.get("__metadata__") or {}
    if "lift" not in meta:
        raise ValueError(f"{path} has no lift metadata: not a CAST artifact")
    return json.loads(meta["lift"])


def adapter_spec(adapter: str) -> VariantSpec:
    """The adapter's registry entry, which must pin its hash."""
    from lift.models.variants import get_spec

    try:
        spec = get_spec(adapter)
    except KeyError:
        raise ValueError(f"{adapter} is not in configs/variants.yaml: after its training job, `lift sync "
                         "pull` "
                         f"and `lift register-adapter {adapter}`, then commit") from None
    if spec.method != "adapter":
        raise ValueError(f"{adapter} is a {spec.method} variant, not an adapter")
    if not _HEX64.fullmatch(str(spec.artifacts.get("adapter", ""))):
        raise ValueError(f"configs/variants.yaml does not pin {adapter}'s hash (artifacts.adapter): register "
                         "it "
                         f"with `lift register-adapter {adapter}` first")
    return spec


def base_reference() -> tuple[str, str]:
    """(Base's id, revision) for ReSta, from configs/project.yaml: a commit sha, or a local directory's
    content hash (tests)."""
    from lift.models.load import resolved_revision

    base_id, rev = config.get("models.base.id"), str(config.get("models.base.revision"))
    if Path(base_id).is_dir():
        return base_id, resolved_revision(base_id, rev)
    if not _SHA.fullmatch(rev):
        raise ValueError(f"models.base.revision is {rev!r}, not a commit sha: pin it (configs/project.yaml) "
                         "before registering ReSta settings")
    return base_id, rev


def cast_params(adapter: str, adapter_hash: str) -> tuple[dict[str, Any], str]:
    """(the params every CAST entry of ``adapter`` copies from its artifact, the artifact's sha256)."""
    from lift.io import sha256_file
    from lift.repair.cast import artifact_path

    path = artifact_path(adapter)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `lift fit-cast --adapter {adapter}` (GPU), then "
                                "`lift sync pull`")
    info = read_cast_info(path)
    if info.get("adapter_hash") != adapter_hash:
        raise ValueError(f"{path} was fitted on adapter hash {str(info.get('adapter_hash'))[:12]}, and "
                         f"configs/variants.yaml pins {adapter_hash[:12]} for {adapter}: fit it again "
                         f"(lift fit-cast --adapter {adapter} --refit)")
    params = {"layer": int(info["condition_layer"]), "threshold": float(info["threshold"]),
              "comparator": info["comparator"], "pool": info["pool"],
              "behavior_layers": [int(x) for x in info["behavior_layers"]]}
    return params, sha256_file(path)


def spec_for(setting: Setting, adapter: str, *, adapter_spec_: VariantSpec | None = None,
             cast: tuple[dict[str, Any], str] | None = None,
             base: tuple[str, str] | None = None) -> VariantSpec:
    """The VariantSpec of one setting on ``adapter``."""
    a = adapter_spec_ or adapter_spec(adapter)
    digest = a.artifacts["adapter"]
    params: dict[str, Any] = {PARAM[setting.method]: round(setting.value, 6)}
    artifacts = {"adapter": digest}
    if setting.method == "resta":
        base_id, base_rev = base or base_reference()
        params.update({"dare": False, "base": base_id, "base_revision": base_rev})
    elif setting.method in ("cast", "cast_always"):
        extra, sha = cast or cast_params(adapter, digest)
        params.update(extra)
        artifacts["cast"] = sha
    return VariantSpec(name=setting.name(adapter), method=setting.method, adapter=adapter, seed=a.seed,
                       params=params, artifacts=artifacts)


def specs_for(adapter: str, chosen: Sequence[Setting]) -> list[VariantSpec]:
    a = adapter_spec(adapter)
    cast = cast_params(adapter, a.artifacts["adapter"]) if any(s.family == "cast" for s in chosen) else None
    base = base_reference() if any(s.method == "resta" for s in chosen) else None
    return [spec_for(s, adapter, adapter_spec_=a, cast=cast, base=base) for s in chosen]


def entry(spec: VariantSpec, note: str = NOTE) -> dict[str, Any]:
    return {**spec.model_dump(), "note": note}


def entry_lines(entry_: Mapping[str, Any], indent: str = "  ") -> list[str]:
    """The entry as YAML block lines; params and artifacts as JSON flow mappings (valid YAML)."""
    inner = indent + "  "
    return [f"{indent}- name: {entry_['name']}", f"{inner}method: {entry_['method']}",
            f"{inner}adapter: {entry_['adapter']}", f"{inner}seed: {int(entry_['seed'])}",
            f"{inner}params: {json.dumps(entry_['params'], sort_keys=True)}",
            f"{inner}artifacts: {json.dumps(entry_['artifacts'], sort_keys=True)}",
            f"{inner}note: {json.dumps(entry_.get('note', NOTE), ensure_ascii=False)}"]


def register(specs: Sequence[VariantSpec], *, registry: Path | None = None,
             replace: bool = False) -> dict[str, str]:
    """Write each spec's entry into the registry (configs/variants.yaml unless ``registry``): {name:
    "added" | "unchanged" | "replaced"}. Another entry under a name is refused unless ``replace`` (each
    output made from the old entry goes stale). The edited text must parse to exactly the old entries plus
    these, or nothing is written."""
    import yaml

    from lift.train import lora

    path = Path(registry) if registry is not None else lora.default_registry()
    text = path.read_text(encoding="utf-8")
    before = lora._registry_specs(yaml.safe_load(text), path)
    outcome: dict[str, str] = {}
    new_text = text
    for spec in specs:
        old = before.get(spec.name)
        if old == spec:
            outcome[spec.name] = "unchanged"
            continue
        if old is not None and not replace:
            raise lora.RegistryConflict(
                f"{lora._shown(path)} already registers {spec.name} differently "
                f"({lora._differs(old, spec)}). "
                "Replacing it changes the cache key of every output made from the registered entry, so those "
                "cells go stale: pass --replace only when that is meant (a CAST artifact fitted again)")
        lines = new_text.splitlines()
        items = [m for line in lines if (m := lora._ITEM.match(line))]
        indent = items[0].group("indent") if items else "  "
        if old is not None:
            span = lora._entry_span(lines, spec.name)
            if span is None:
                raise ValueError(f"{path}: cannot find {spec.name}'s entry to replace; replace it by hand")
            start, end = span
            new_text = "\n".join(lines[:start] + entry_lines(entry(spec), indent) + lines[end:]) + "\n"
            outcome[spec.name] = "replaced"
        else:
            body = new_text if not new_text or new_text.endswith("\n") else new_text + "\n"
            new_text = body + "\n".join(entry_lines(entry(spec), indent)) + "\n"
            outcome[spec.name] = "added"
    if new_text == text:
        return outcome
    expected = {**before, **{s.name: s for s in specs}}
    try:
        after = lora._registry_specs(yaml.safe_load(new_text), path)
    except (yaml.YAMLError, ValueError):
        after = None
    if after != expected:
        raise ValueError(f"editing {path} as text would not leave exactly its entries plus the repair "
                         "settings "
                         "(its layout is not the one this writer expects); add them by hand")
    lora._write_text(path, new_text)
    if path.resolve() == lora.default_registry().resolve():
        clear = getattr(config.load, "cache_clear", None)
        if clear is not None:
            clear()
    return outcome
