"""Surface parity: every public item of the ported Rust modules has a Cyrius counterpart.

Port plan §6 (a per-milestone gate) and M8 (the release gate). Copied from impetus's scripts/surface_parity.py
(itself kimiya's and tanmatra's) with pravash's names. Usage, from the repo root:
    python3 -I scripts/surface_parity.py .

For each ported module it lists rust-old/src/<module>.rs's public items before `#[cfg(test)]` (and skips a fn whose
own attributes include `#[cfg(test)]`, a test helper) and checks that src/ defines the Cyrius name the naming rules
give (port plan §7.4):
- free `pub fn f`                         -> `fn f(`
- `impl Struct { pub fn f(...) }`, no self -> `fn Pv<Struct>_f(`
- `impl Struct { pub fn m(&self ...) }`    -> `fn m(` inside `impl Pv<Struct> {`
- `impl Enum { pub fn m(self ...) }`       -> `fn <snake(Enum)>_m(`
- `impl Default for X`                     -> `fn Pv<X>_default(`
- `pub const X` inside `impl Owner`        -> `PV_<SNAKE(Owner)>_X`, a const or a global (`FluidMaterial::WATER` is
                                              the struct global PV_FLUID_MATERIAL_WATER); a free `pub const X` ->
                                              `PV_X`
- `pub enum E` and each variant V          -> `enum <ENUM_NAMES[E]> {` and `<ENUM_PREFIX[E]><SNAKE(V)> =`
- `pub struct S` and each `pub` field f    -> `struct Pv<S> {` with a field f, or FIELD_RENAMES' fields
A function the naming rules would name otherwise is in FN_RENAMES; one whose Cyrius owner arrives with a later
milestone is in PENDING_FNS (counted and listed); anything else needs an EXCEPTIONS entry with its reason. Every file
under rust-old/src must be in MODULES, PENDING_MODULES (a later milestone), DEFERRED_MODULES or
NOT_APPLICABLE_MODULES. It also checks that tests/lib.cyr includes exactly cyrius.cyml's [lib] list, and scans for the
cyrius 6.7.18 field-store miscompile (field_store_scan). Exits 1 on a missing item, an EXCEPTIONS entry that names no
Rust item, or a file unaccounted for.
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

MODULES = ["error", "common"]
PENDING_MODULES = {
    "buoyancy": "M1 part 4", "vortex": "M1 part 4", "bridge": "M1 part 4",
    "shallow": "M2", "grid": "M3", "vof": "M3", "phase_field": "M3", "sph": "M4–M5",
    "coupling": "M6", "mpm": "M6", "compute": "M7", "logging": "M7", "ai": "M7",
}
DEFERRED_MODULES = {}
NOT_APPLICABLE_MODULES = {
    "lib": "crate root: module list and re-exports",
    "oracle": "the hidden oracle hooks, Rust-only port tooling",
    **{f"{m}/oracle": "oracle hooks" for m in ("common", "coupling", "grid", "mpm", "phase_field", "shallow",
                                               "sph", "vof")},
}

# A payload enum is a kind enum plus one struct carrying every variant's fields (port plan §7.3).
ENUM_NAMES = {"PravashError": "PvError"}
ENUM_PREFIX = {"PravashError": "PV_ERR_"}
# Rust function -> the Cyrius function that provides it, where the naming rules give another name.
FN_RENAMES = {}
# Rust field -> the Cyrius fields that carry it.
FIELD_RENAMES = {}
PENDING_CONSTS = {}
# Rust functions whose Cyrius owner arrives with a later milestone: counted, listed, never silently dropped.
PENDING_FNS = {}
EXCEPTIONS = {
    "error::type Result": "the stdlib's value-form Result, whose Err payload is a PV_ERR_* code (ADR 0002 #1)",
}


def snake(camel):
    out = []
    for k, ch in enumerate(camel):
        if ch.isupper() and k > 0:
            prev = camel[k - 1]
            nxt = camel[k + 1] if k + 1 < len(camel) else ""
            if prev.islower() or prev.isdigit() or (prev.isupper() and nxt.islower()):
                out.append("_")
        out.append(ch.upper())
    return "".join(out)


def library_part(text):
    cut = text.find("#[cfg(test)]\nmod ")
    return text if cut < 0 else text[:cut]


def cyrius_index():
    """Every fn, impl method (as Type_m), const, enum member, enum and struct field in src/."""
    fns, consts, members, enums, structs = set(), set(), set(), set(), {}
    globals_ = set()
    for path in sorted((root / "src").glob("*.cyr")):
        impl = None
        depth = 0
        for line in path.read_text().splitlines():
            s = line.strip()
            m = re.match(r"impl (\w+) \{", s)
            if m and depth == 0:
                impl, depth = m.group(1), 0
            m = re.match(r"fn (\w+)\(", s)
            if m:
                fns.add(f"{impl}::{m.group(1)}" if impl else m.group(1))
            m = re.match(r"const (\w+) =", s)
            if m:
                consts.add(m.group(1))
            m = re.match(r"var (PV_\w+)(?::| =)", line)
            if m:
                globals_.add(m.group(1))
            m = re.match(r"(PV_\w+) = -?\d+;", s)
            if m:
                members.add(m.group(1))
            m = re.match(r"enum (\w+) \{", s)
            if m:
                enums.add(m.group(1))
            m = re.match(r"struct (\w+) \{(.*)\}", s)
            if m:
                structs[m.group(1)] = {f.split(":")[0].strip() for f in m.group(2).split(";") if f.strip()}
            elif (m := re.match(r"struct (\w+) \{", s)):
                structs[m.group(1)] = set()
                current = m.group(1)
                continue
            if impl is not None:
                depth += line.count("{") - line.count("}")
                if depth <= 0 and s == "}":
                    impl = None
    # Multi-line struct bodies.
    for path in sorted((root / "src").glob("*.cyr")):
        text = path.read_text()
        for m in re.finditer(r"struct (\w+) \{\n(.*?)\n\}", text, re.S):
            structs[m.group(1)] = {l.strip().split(":")[0].rstrip(";").strip()
                                   for l in m.group(2).splitlines() if l.strip() and not l.strip().startswith("#")}
    return fns, consts | globals_, members, enums, structs


VIS = r"pub(?:\((?:crate|super)\))?"


def rust_items(module):
    """(kind, owner, name, extra) for each public item of rust-old/src/<module>.rs."""
    text = library_part((root / "rust-old/src" / f"{module}.rs").read_text())
    items = []
    owner, owner_kind, trait = None, None, None
    enum_name = None
    struct_name = None
    lines = text.splitlines()
    for li, line in enumerate(lines):
        s = line.strip()
        if (m := re.match(r"impl (\w+(?:::\w+)*) for (\w+)", s)):
            owner, trait = m.group(2), m.group(1).split("::")[-1]
            items.append(("trait", owner, trait, None))
            continue
        if (m := re.match(r"impl(?:<[^>]*>)? (\w+)(?:<[^>]*>)? \{", s)):
            owner, trait = m.group(1), None
            continue
        if (m := re.match(VIS + r" enum (\w+)", s)):
            enum_name, struct_name = m.group(1), None
            items.append(("enum", None, enum_name, None))
            continue
        if (m := re.match(VIS + r" struct (\w+)", s)):
            struct_name, enum_name = m.group(1), None
            items.append(("struct", None, struct_name, None))
            continue
        if s.startswith("}") and not line.startswith(" "):
            owner, enum_name, struct_name = None, None, None
            continue
        if enum_name and (m := re.match(r"([A-Z]\w*)(?:\(|,|\s*\{|$)", s)) and not s.startswith("#"):
            items.append(("variant", enum_name, m.group(1), None))
        if struct_name and (m := re.match(r"pub (\w+):", s)):
            items.append(("field", struct_name, m.group(1), None))
        if (m := re.match(VIS + r" fn (\w+)(?:<[^>]*>)?\((&?(?:mut )?self)?", s)):
            # A test-only helper (`#[cfg(test)]` among the attributes above it) is not library surface.
            back = li - 1
            test_only = False
            while back >= 0 and lines[back].strip().startswith("#["):
                test_only = test_only or lines[back].strip() == "#[cfg(test)]"
                back -= 1
            if test_only:
                continue
            has_self = bool(m.group(2))
            # A signature that breaks after `(` names its receiver on the next line.
            if not has_self and s.endswith("(") and li + 1 < len(lines):
                has_self = bool(re.match(r"&?(?:mut )?self\b", lines[li + 1].strip()))
            items.append(("fn", owner, m.group(1), has_self))
        if (m := re.match(r"pub const (\w+):", s)):
            items.append(("const", owner, m.group(1), None))
        if (m := re.match(r"pub\((?:crate|super)\) const (\w+):", s)):
            items.append(("pconst", owner, m.group(1), None))
        if (m := re.match(VIS + r" type (\w+)", s)):
            items.append(("type", None, m.group(1), None))
    return items


def profile_drift():
    """tests/lib.cyr includes exactly cyrius.cyml's [lib] module list, in the same order."""
    manifest = (root / "cyrius.cyml").read_text()
    m = re.search(r"^\[lib\]\nmodules = \[(.*?)\]", manifest, re.S | re.M)
    if m is None:
        return ["cyrius.cyml has no [lib] modules list"]
    want = re.findall(r'"([^"]+)"', m.group(1))
    got = re.findall(r'^include "([^"]+)"', (root / "tests/lib.cyr").read_text(), re.M)
    if got != want:
        return [f"tests/lib.cyr includes {len(got)} modules that differ from cyrius.cyml [lib]'s {len(want)}"]
    return []


def field_store_scan():
    """No `s.f = g(.., t.e, ..)` where g returns a struct and t.e is struct-typed: cyrius 6.7.18 stores t.e instead of
    g's result (cyrius docs/development/issues/2026-11-09-field-store-of-struct-call-copies-a-field-argument.md). Bind
    the call to a local first. Types are resolved from struct fields, fn parameters, `var x: T` and `self` in an impl."""
    files = sorted(list((root / "src").glob("*.cyr")) + list((root / "tests").glob("*.*cyr")))
    texts = {f: f.read_text() for f in files}
    fields = {}
    for text in texts.values():
        for m in re.finditer(r"^struct (\w+)\s*\{(.*?)\}", text, re.M | re.S):
            fields[m.group(1)] = dict(re.findall(r"(\w+)\s*:\s*\*?(\w+)\s*;", m.group(2)))
    sret = set()
    for text in texts.values():
        for m in re.finditer(r"^\s*fn (\w+)\([^)]*\)\s*:\s*(\w+)\s*\{", text, re.M):
            if m.group(2) in fields:
                sret.add(m.group(1))
    store = re.compile(r"([a-z_]\w*(?:\.\w+)+)\s*=\s*(\w+)\(([^;]*)\);")
    ref = re.compile(r"\b([a-z_]\w*)((?:\.[a-z_]\w*)+)\b(?!\s*\()")
    out = []
    for f, text in texts.items():
        types, impl = {}, None
        for i, line in enumerate(text.split("\n"), 1):
            m = re.match(r"^impl (\w+)", line)
            if m:
                impl = m.group(1)
            elif line.startswith("}"):
                impl = None
            m = re.match(r"^\s*fn \w+\(([^)]*)\)", line)
            if m:
                types = dict(re.findall(r"(\w+)\s*:\s*\*?(\w+)", m.group(1)))
                if impl:
                    types["self"] = impl
            for name, ty in re.findall(r"\bvar (\w+)\s*:\s*\*?(\w+)", line):
                types[name] = ty
            for sm in store.finditer(line):
                if sm.group(2) not in sret:
                    continue
                for base, chain in ref.findall(sm.group(3)):
                    ty = types.get(base)
                    for name in chain.split(".")[1:]:
                        ty = fields.get(ty, {}).get(name)
                    if ty in fields:
                        out.append(f"{f.relative_to(root)}:{i}: a struct call with a struct-field argument stored into "
                                   f"a field (cyrius 6.7.18 misstores it; bind to a local): {line.strip()[:90]}")
                        break
    return out


def main():
    fns, consts, members, enums, structs = cyrius_index()
    enum_owners = set()
    missing, ok = [], 0
    used_exceptions = set()
    used_renames = set()
    used_pending = set()
    pending = []
    for module in MODULES:
        items = rust_items(module)
        enum_owners |= {name for kind, _, name, _ in items if kind == "enum"}
        for kind, owner, name, extra in items:
            key = f"{module}::{kind} {name}" if kind in ("type",) else None
            label = f"{module}::{owner + '::' if owner else ''}{name} ({kind})"
            if kind == "type":
                ex = f"{module}::type {name}"
                if ex in EXCEPTIONS:
                    used_exceptions.add(ex)
                    ok += 1
                else:
                    missing.append(label)
                continue
            if kind == "trait":
                ex = f"{module}::impl {name} for {owner}"
                if ex in EXCEPTIONS:
                    used_exceptions.add(ex)
                    ok += 1
                elif name == "Default" and f"Pv{owner}_default" in fns:
                    ok += 1
                else:
                    missing.append(label)
                continue
            ex = f"{module}::{owner + '::' if owner else ''}{name}"
            if ex in EXCEPTIONS:
                used_exceptions.add(ex)
                ok += 1
                continue
            if kind == "enum":
                want = ENUM_NAMES.get(name, f"Pv{name}")
                (ok := ok + 1) if want in enums else missing.append(f"{label} -> enum {want}")
                continue
            if kind == "variant":
                want = ENUM_PREFIX.get(owner, f"PV_{snake(owner)}_") + snake(name)
                (ok := ok + 1) if want in members else missing.append(f"{label} -> {want}")
                continue
            if kind == "struct":
                (ok := ok + 1) if f"Pv{name}" in structs else missing.append(f"{label} -> struct Pv{name}")
                continue
            if kind == "field":
                want = FIELD_RENAMES.get((owner, name), [name])
                have = structs.get(f"Pv{owner}", set())
                (ok := ok + 1) if all(w in have for w in want) else missing.append(f"{label} -> fields {want}")
                continue
            if kind == "pconst":
                want = "_PV_" + name
                (ok := ok + 1) if want in consts else missing.append(f"{label} -> {want}")
                continue
            if kind == "const":
                if name in PENDING_CONSTS:
                    ok += 1
                    continue
                want = f"PV_{snake(owner)}_{name}" if owner else f"PV_{name}"
                (ok := ok + 1) if want in consts else missing.append(f"{label} -> {want}")
                continue
            if kind == "fn":
                if f"{owner + '::' if owner else ''}{name}" in PENDING_FNS:
                    used_pending.add(f"{owner + '::' if owner else ''}{name}")
                    pending.append(label)
                    ok += 1
                    continue
                rename = FN_RENAMES.get(f"{owner + '::' if owner else ''}{name}")
                if rename is not None:
                    used_renames.add(f"{owner + '::' if owner else ''}{name}")
                    (ok := ok + 1) if rename in fns else missing.append(f"{label} -> {rename}")
                    continue
                if owner is None:
                    want = [name]
                elif owner in enum_owners:
                    want = [f"{snake(owner).lower()}_{name}"]
                elif extra:
                    want = [f"Pv{owner}::{name}"]
                else:
                    want = [f"Pv{owner}_{name}"]
                (ok := ok + 1) if any(w in fns for w in want) else missing.append(f"{label} -> {want[0]}")
    for ex in EXCEPTIONS:
        if ex not in used_exceptions:
            missing.append(f"EXCEPTIONS entry names no Rust item: {ex}")
    for rn in FN_RENAMES:
        if rn not in used_renames:
            missing.append(f"FN_RENAMES entry names no Rust function: {rn}")
    for pn in PENDING_FNS:
        if pn not in used_pending:
            missing.append(f"PENDING_FNS entry names no Rust function: {pn}")
    missing.extend(profile_drift())
    missing.extend(field_store_scan())
    # Every rust-old/src file is accounted for.
    accounted = set(MODULES) | set(PENDING_MODULES) | set(DEFERRED_MODULES) | set(NOT_APPLICABLE_MODULES)
    for path in sorted((root / "rust-old/src").rglob("*.rs")):
        rel = str(path.relative_to(root / "rust-old/src"))[:-3]
        if rel not in accounted:
            missing.append(f"rust-old/src/{rel}.rs is in no module list")
    print(f"surface parity ({', '.join(MODULES)}): {ok} items ok ({len(pending)} pending a later "
          f"milestone), {len(missing)} missing")
    for p in pending:
        print("  PENDING", p)
    for m in missing:
        print("  MISSING", m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
