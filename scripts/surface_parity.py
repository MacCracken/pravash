"""Surface parity: every public item of the ported Rust modules has a Cyrius counterpart.

Port plan §6 (a per-milestone gate) and M8 (the release gate). Copied from impetus's scripts/surface_parity.py
(itself kimiya's and tanmatra's) with pravash's names. Usage, from the repo root:
    python3 -I -B scripts/surface_parity.py . [--extra]

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
- `pub trait T` and each of its methods m  -> `trait Pv<T> {` declaring `fn m(`
- `impl Debug for S` (hand-written)        -> `fn debug(` inside `impl Pv<S> {`
- each `#[derive(..)]` trait               -> the function the derive rules give (the comment above
                                              DERIVE_RENAMES: `to_json`, `_from_json`, `_default`, `_clone`, `eq`,
                                              `debug`), or a plain value (Copy; an enum's PartialEq)
A function the naming rules would name otherwise is in FN_RENAMES; one whose Cyrius owner arrives with a later
milestone is in PENDING_FNS, and a type arriving with a later part of its module (its fields and methods with it) is
in PENDING_TYPES (both counted and listed); anything else needs an EXCEPTIONS entry with its reason.

The reverse direction (M8 part c, the full surface diff): every public name the port defines (no `_` prefix: a fn,
an impl method, a struct, an enum, a trait, a const, a global, an enum member) in the [lib] modules and OUTSIDE_LIB is
either a counterpart the forward check found or matches a CYRIUS_ONLY rule, which gives the ADR 0002 row or plan
section that admits it. `--extra` lists each rule's names.

Every file under rust-old/src must be in MODULES, PENDING_MODULES (a later milestone), DEFERRED_MODULES or
NOT_APPLICABLE_MODULES. It also checks that tests/lib.cyr includes exactly cyrius.cyml's [lib] list, and scans for
the cyrius 6.7.18 field-store miscompile (field_store_scan). Exits 1 on a missing item, a public Cyrius name no rule
explains, a table entry that names nothing (an EXCEPTIONS, DERIVE_*, FN_RENAMES or PENDING_* entry, or a CYRIUS_ONLY
rule that matches no name), or a file unaccounted for.
"""
import re
import sys
from pathlib import Path

root = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), ".")).resolve()

MODULES = ["error", "common", "buoyancy", "vortex", "bridge", "shallow", "grid", "vof", "phase_field", "sph",
           "coupling", "mpm", "compute", "ai", "logging"]
PENDING_MODULES = {}
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
FN_RENAMES = {
    # sph's free `step`: one namespace serves pravash and its consumers, and a bare `step` is the likeliest to
    # collide (ADR 0002 #17).
    "step": "sph_step",
    # logging's free functions: bare `init` names collide first of all (ADR 0002 #28).
    "init": "logging_init",
    "init_with_level": "logging_init_with_level",
    "init_profiling": "logging_init_profiling",
}
# Rust field -> the Cyrius fields that carry it.
FIELD_RENAMES = {
    # `[f32; 3]`: a tuple element can't be an f32 and a struct holds no array, so three fields (ADR 0002 #26).
    ("SphKernelParams", "gravity"): ["gravity_x", "gravity_y", "gravity_z"],
}
PENDING_CONSTS = {}
# Rust functions whose Cyrius owner arrives with a later milestone: counted, listed, never silently dropped.
PENDING_FNS = {}
# Rust types (with their fields and methods) that arrive with a later part of a ported module: counted and listed.
PENDING_TYPES = {}
EXCEPTIONS = {
    "buoyancy::DragCoefficient": "a unit struct that only holds the drag coefficients: PV_DRAG_COEFFICIENT_* "
                                 "(src/constants.cyr); its serde form is drag_coefficient_to_json / _from_json",
    "error::type Result": "the stdlib's value-form Result, whose Err payload is a PV_ERR_* code (ADR 0002 #1)",
    "coupling::AddedMassCoefficient": "a unit struct that only holds the added-mass coefficients: "
                                      "PV_ADDED_MASS_COEFFICIENT_* (src/constants.cyr)",
    "compute::KernelDerivatives": "a unit struct that only namespaces its associated functions: "
                                  "PvKernelDerivatives_* (src/compute.cyr; autodiff_gradient in src/compute_ad.cyr, "
                                  "outside the bundle, ADR 0002 #26)",
}


# Rust's derives, read from `#[derive(..)]` (port plan §7.4). Copy, and Clone on a Copy type, are a plain Cyrius
# value copied by assignment; PartialEq and Eq on an enum compare its words with `==`. Otherwise a derive is a
# function: Serialize `Pv<T>::to_json` (an enum's `<snake>_to_json`), Deserialize `Pv<T>_from_json`
# (`<snake>_from_json`), Default `Pv<T>_default`, Clone `Pv<T>_clone`, PartialEq and Eq `Pv<T>::eq`, Debug
# `Pv<T>::debug` (an enum's `<snake>_debug`) and thiserror's Error `pv_error_message`. A derive the rules name otherwise is in DERIVE_RENAMES;
# one that waits on a later part is in PENDING_DERIVES (counted and listed); any other needs a DERIVE_EXCEPTIONS
# entry with its reason.
DERIVE_RENAMES = {
    # A unit struct of coefficients (EXCEPTIONS above): its serde form and its Debug are free functions.
    ("DragCoefficient", "Serialize"): "drag_coefficient_to_json",
    ("DragCoefficient", "Deserialize"): "drag_coefficient_from_json",
    ("DragCoefficient", "Debug"): "drag_coefficient_debug",
}
DERIVE_EXCEPTIONS = {
    "buoyancy::derive PartialEq for DragCoefficient": "a unit struct: its one value equals itself, and the port has "
                                                      "no value of it, only the PV_DRAG_COEFFICIENT_* constants",
    "coupling::derive Clone for RigidBody": "no storage of its own (three PvVec3, a mass and a PvShape): a struct "
                                            "assignment copies it, as Rust's Clone does",
    "error::derive Debug for PravashError": "a 2.0 error is a PV_ERR_* code with no payload (ADR 0002 #1), so the "
                                            "variant's fields Debug prints aren't there; pv_error_name(code) gives "
                                            "the variant's name",
}
# Derives a later part ports, by trait: counted and listed, never silently dropped.
PENDING_DERIVES = {}
# Public Cyrius names with no Rust item: (a regex over the name, why the port has it). Each rule must match a name.
CYRIUS_ONLY = [
    (r"Pv\w+_drop", "Rust's Drop, made explicit (ADR 0002 #9)"),
    (r"PvColumn(_new|::\w+)?", "Rust's public `Vec<f64>` and `Vec<usize>` fields: a resizable column of 8-byte words "
                               "(ADR 0002 #11)"),
    (r"PvF32Column(::\w+)?", "Rust's `Vec<f32>` buffers: packed singles a backend uploads as they are (ADR 0002 #26)"),
    (r"PvVec3(_new|_zero|::\w+)?", "glam's `DVec3` (through hisab) in public fields and signatures: three f64 and the "
                                   "operations pravash calls (port plan §3.1, M1)"),
    (r"PV_ERR_(NONE|DESERIALIZE|OUT_OF_MEMORY)",
     "the Ok code, and the Err codes for a refused JSON document and a failed allocation, where Rust has "
     "serde_json's error and aborts (ADR 0002 #1, #10, #11)"),
    (r"(pv_error|flow_regime|boundary_condition|body_shape|constitutive_model|compute_op)_name",
     "a variant's Rust name (`\"Laminar\"`), the text serde and Debug print for it"),
    (r"PV_F32_EPSILON|PV_F64_(INFINITY|NEG_INFINITY|NAN|MAX|PI)", "Rust's `f32::EPSILON`, `f64::INFINITY`, `f64::NAN`, "
                                                                 "`f64::MAX` and `PI`, generated: a Cyrius literal "
                                                                 "has no NaN or infinity"),
    (r"PV_FLUID_MATERIAL_[A-Z]+_(DENSITY|VISCOSITY|SURFACE_TENSION|SPEED_OF_SOUND)",
     "the fields of the FluidMaterial presets, generated; the struct globals PV_FLUID_MATERIAL_WATER … are built "
     "from them"),
    (r"Pv(FluidGrid|SphSolver)::set_threads", "the runtime thread count that replaces the `parallel` feature "
                                              "(ADR 0002 #4)"),
    (r"PvSphSolver_init", "a solver built in place, allocating nothing (ADR 0002 #19)"),
    (r"PvNeighborGrid(_init|_build)?",
     "the cells-only neighbour grid a caller passes to `drag_from_particles_indexed`, Rust's `&SpatialHash` "
     "(ADR 0002 #16, #23)"),
    (r"PvNeighborLists", "Rust's `(&[u32], &[usize])` from `SphSolver::neighbor_lists` (ADR 0002 #19)"),
    (r"PvShape(_sphere|_box)?|PvConstitutiveLaw(_\w+)?|PvNonNewtonianModel(_\w+)?",
     "the payload struct of a data-carrying enum and its variant constructors (port plan §7.3; ADR 0002 #15, #23, "
     "#25)"),
    (r"PvMat2(_new)?", "Rust's `[f64; 4]` 2 × 2 matrices in MpmParticle (ADR 0002 #25)"),
    (r"PvReaction", "Rust's `(f64, f64)` from `ReactionProvider::react`: a call through a `dyn` returns one value "
                    "(ADR 0002 #18)"),
    (r"PvFlushGuard(::flush)?", "tracing-chrome's FlushGuard, which `init_profiling` returns (ADR 0002 #28)"),
    (r"PvDaimonConfig::(set_api_key|clear_api_key|set_endpoint)|PvHooshConfig::set_endpoint",
     "the setters of the String fields, which copy into owned storage (ADR 0002 #27)"),
    (r"PvMultiPhaseConfig::(phase_count|push_phase)", "Rust's `phases.len()` and `phases.push(..)` (ADR 0002 #22)"),
]
# Files outside the [lib] bundle whose public names are part of the port.
OUTSIDE_LIB = {"src/compute_ad.cyr"}


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


def cyrius_traits():
    """The traits src/ declares, its (trait, type) impls, and each trait's methods as `Trait::m`."""
    traits, impls, trait_fns = set(), set(), set()
    for path in sorted((root / "src").glob("*.cyr")):
        text = path.read_text()
        traits |= set(re.findall(r"^trait (\w+) \{", text, re.M))
        impls |= set(re.findall(r"^impl (\w+) for (\w+) \{", text, re.M))
        for m in re.finditer(r"^trait (\w+) \{\n(.*?)^\}", text, re.M | re.S):
            trait_fns |= {f"{m.group(1)}::{f}" for f in re.findall(r"^\s+fn (\w+)\(", m.group(2), re.M)}
    return traits, impls, trait_fns


def rust_derives(module):
    """(type, is_enum, [derived traits]) for each public struct and enum of rust-old/src/<module>.rs, read from the
    `#[derive(..)]` attributes above it (doc comments and other attributes may interleave)."""
    lines = library_part((root / "rust-old/src" / f"{module}.rs").read_text()).splitlines()
    out = []
    for li, line in enumerate(lines):
        m = re.match(VIS + r" (struct|enum) (\w+)", line)
        if not m:
            continue
        traits = []
        back = li - 1
        while back >= 0 and lines[back].strip().startswith(("#[", "///", "//")):
            for d in re.findall(r"derive\(([^)]*)\)", lines[back]):
                traits += [t.strip() for t in d.split(",") if t.strip()]
            back -= 1
        out.append((m.group(2), m.group(1) == "enum", traits))
    return out


def cyrius_public():
    """{name: file} for every public name the port defines: the [lib] modules and the files outside the bundle
    (OUTSIDE_LIB). A fn is `f`, or `Type::m` inside `impl Type {`; a struct, an enum, a trait, a const, a global and
    an enum member by its name. A trait's methods and those of an `impl Trait for Type` are the trait's (the forward
    check reads them there), and a `_`-prefixed name is private."""
    manifest = (root / "cyrius.cyml").read_text()
    lib = re.findall(r'"(src/[^"]+)"', re.search(r"^\[lib\]\nmodules = \[(.*?)\]", manifest, re.S | re.M).group(1))
    out = {}
    for rel in lib + sorted(OUTSIDE_LIB):
        impl = None
        for line in (root / rel).read_text().splitlines():
            if (m := re.match(r"(impl|trait) (\w+)( for \w+)? \{", line)):
                impl = None if m.group(1) == "trait" or m.group(3) else m.group(2)
                if m.group(1) == "trait":
                    out[m.group(2)] = rel
                continue
            if line.startswith("}"):
                impl = None
                continue
            s = line.strip()
            if line.startswith(" ") and (m := re.match(r"fn (\w+)\(", s)):
                if impl is not None and not m.group(1).startswith("_"):
                    out[f"{impl}::{m.group(1)}"] = rel
            elif (m := re.match(r"fn (\w+)\(", line)) and not m.group(1).startswith("_"):
                out[m.group(1)] = rel
            elif (m := re.match(r"(?:struct|enum) (\w+)", line)) and not m.group(1).startswith("_"):
                out[m.group(1)] = rel
            elif (m := re.match(r"(?:const|var) (\w+)", line)) and not m.group(1).startswith("_"):
                out[m.group(1)] = rel
            elif (m := re.match(r"\s+(PV_\w+) = -?\d+", line)):
                out[m.group(1)] = rel
    return out


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
    trait_name = None
    for li, line in enumerate(lines):
        s = line.strip()
        if trait_name is not None:
            # A trait's methods carry no `pub`; the body ends at the column-0 `}`.
            if line.startswith("}"):
                trait_name = None
            elif (m := re.match(r"fn (\w+)", s)):
                items.append(("trait fn", trait_name, m.group(1), None))
            continue
        if (m := re.match(r"impl (\w+(?:::\w+)*) for (\w+)", s)):
            owner, trait = m.group(2), m.group(1).split("::")[-1]
            items.append(("trait", owner, trait, None))
            continue
        if (m := re.match(r"impl(?:<[^>]*>)? (\w+)(?:<[^>]*>)? \{", s)):
            owner, trait = m.group(1), None
            continue
        if (m := re.match(VIS + r" trait (\w+)", s)):
            items.append(("trait decl", None, m.group(1), None))
            trait_name = m.group(1)
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
    traits, impls, trait_fns = cyrius_traits()
    enum_owners = set()
    missing, ok = [], 0
    # The Cyrius names the forward check finds: what the reverse check (every public Cyrius name) starts from.
    matched = set()
    used_exceptions = set()
    used_renames = set()
    used_pending = set()
    used_derive = set()
    pending = []
    derived = 0

    def have(want, pool):
        """The first name of `want` that `pool` holds, recorded as matched; None when there is none."""
        for w in want:
            if w in pool:
                matched.add(w)
                return w
        return None

    for module in MODULES:
        items = rust_items(module)
        enum_owners |= {name for kind, _, name, _ in items if kind == "enum"}
        for kind, owner, name, extra in items:
            label = f"{module}::{owner + '::' if owner else ''}{name} ({kind})"
            # A type arriving with a later part, with its fields, variants, methods and trait impls.
            tkey = f"{module}::{name if kind in ('struct', 'enum') and not owner else owner}"
            if (owner or kind in ("struct", "enum")) and tkey in PENDING_TYPES:
                used_pending.add(tkey)
                pending.append(f"{label} [{PENDING_TYPES[tkey]}]")
                ok += 1
                continue
            if kind == "type":
                ex = f"{module}::type {name}"
                if ex in EXCEPTIONS:
                    used_exceptions.add(ex)
                    ok += 1
                else:
                    missing.append(label)
                continue
            if kind == "trait decl":
                # `pub trait X` is `trait PvX` (a Rust `&dyn X` is a `dyn PvX`).
                (ok := ok + 1) if have([f"Pv{name}"], traits) else missing.append(f"{label} -> trait Pv{name}")
                continue
            if kind == "trait fn":
                want = f"Pv{owner}::{name}"
                (ok := ok + 1) if want in trait_fns else missing.append(f"{label} -> {want}")
                continue
            if kind == "trait":
                ex = f"{module}::impl {name} for {owner}"
                if ex in EXCEPTIONS:
                    used_exceptions.add(ex)
                    ok += 1
                elif name == "Default" and have([f"Pv{owner}_default"], fns):
                    ok += 1
                elif name == "Debug" and have([f"Pv{owner}::debug"], fns):
                    # A hand-written Debug is `debug(sb, alternate)`: `{:?}` and `{:#?}` (ADR 0002 #27).
                    ok += 1
                elif (f"Pv{name}", f"Pv{owner}") in impls:
                    ok += 1
                else:
                    want = {"Default": f"Pv{owner}_default", "Debug": f"Pv{owner}::debug"}
                    missing.append(f"{label} -> {want.get(name, f'impl Pv{name} for Pv{owner}')}")
                continue
            ex = f"{module}::{owner + '::' if owner else ''}{name}"
            if ex in EXCEPTIONS:
                used_exceptions.add(ex)
                ok += 1
                continue
            if kind == "enum":
                want = ENUM_NAMES.get(name, f"Pv{name}")
                (ok := ok + 1) if have([want], enums) else missing.append(f"{label} -> enum {want}")
                continue
            if kind == "variant":
                want = ENUM_PREFIX.get(owner, f"PV_{snake(owner)}_") + snake(name)
                (ok := ok + 1) if have([want], members) else missing.append(f"{label} -> {want}")
                continue
            if kind == "struct":
                (ok := ok + 1) if have([f"Pv{name}"], structs) else missing.append(f"{label} -> struct Pv{name}")
                continue
            if kind == "field":
                want = FIELD_RENAMES.get((owner, name), [name])
                have_fields = structs.get(f"Pv{owner}", set())
                (ok := ok + 1) if all(w in have_fields for w in want) else missing.append(f"{label} -> fields {want}")
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
                (ok := ok + 1) if have([want], consts) else missing.append(f"{label} -> {want}")
                continue
            if kind == "fn":
                qual = f"{owner + '::' if owner else ''}{name}"
                if qual in PENDING_FNS:
                    used_pending.add(qual)
                    pending.append(f"{label} [{PENDING_FNS[qual]}]")
                    ok += 1
                    continue
                rename = FN_RENAMES.get(qual)
                if rename is not None:
                    used_renames.add(qual)
                    (ok := ok + 1) if have([rename], fns) else missing.append(f"{label} -> {rename}")
                    continue
                if owner is None:
                    want = [name]
                elif owner in enum_owners:
                    want = [f"{snake(owner).lower()}_{name}"]
                elif extra:
                    want = [f"Pv{owner}::{name}"]
                else:
                    want = [f"Pv{owner}_{name}"]
                (ok := ok + 1) if have(want, fns) else missing.append(f"{label} -> {want[0]}")
        # The derived traits (port plan §7.4): each one a Cyrius function, a value copy, or an entry with a reason.
        for tname, is_enum, derives in rust_derives(module):
            for tr in derives:
                key = f"{module}::derive {tr} for {tname}"
                label = f"{module}::{tname} (derive {tr})"
                if key in DERIVE_EXCEPTIONS:
                    used_derive.add(key)
                    ok += 1
                    derived += 1
                    continue
                if tr == "Copy" or (tr == "Clone" and "Copy" in derives) or (tr in ("PartialEq", "Eq") and is_enum):
                    # A Copy type is a plain Cyrius value, copied by assignment; an enum value is a word, compared
                    # with `==`.
                    ok += 1
                    derived += 1
                    continue
                rename = DERIVE_RENAMES.get((tname, tr))
                if rename is not None:
                    used_derive.add((tname, tr))
                    if have([rename], fns):
                        ok += 1
                        derived += 1
                    else:
                        missing.append(f"{label} -> {rename}")
                    continue
                stem = snake(tname).lower()
                want = {
                    "Serialize": [f"{stem}_to_json"] if is_enum else [f"Pv{tname}::to_json"],
                    "Deserialize": [f"{stem}_from_json"] if is_enum else [f"Pv{tname}_from_json"],
                    "Default": [f"Pv{tname}_default"],
                    "Clone": [f"Pv{tname}_clone"],
                    "PartialEq": [f"Pv{tname}::eq"],
                    "Eq": [f"Pv{tname}::eq"],
                    "Debug": [f"{stem}_debug"] if is_enum else [f"Pv{tname}::debug"],
                    "thiserror::Error": ["pv_error_message"],
                }.get(tr)
                if want is None:
                    missing.append(f"{label}: no rule for this derive")
                elif have(want, fns):
                    ok += 1
                    derived += 1
                elif tr in PENDING_DERIVES:
                    pending.append(f"{label} [{PENDING_DERIVES[tr]}]")
                    used_derive.add(tr)
                    ok += 1
                else:
                    missing.append(f"{label} -> {want[0]}")
    for ex in EXCEPTIONS:
        if ex not in used_exceptions:
            missing.append(f"EXCEPTIONS entry names no Rust item: {ex}")
    for ex in list(DERIVE_EXCEPTIONS) + list(DERIVE_RENAMES) + list(PENDING_DERIVES):
        if ex not in used_derive:
            missing.append(f"derive entry names no Rust derive left to it: {ex}")
    for rn in FN_RENAMES:
        if rn not in used_renames:
            missing.append(f"FN_RENAMES entry names no Rust function: {rn}")
    for pn in PENDING_FNS:
        if pn not in used_pending:
            missing.append(f"PENDING_FNS entry names no Rust function: {pn}")
    for pt in PENDING_TYPES:
        if pt not in used_pending:
            missing.append(f"PENDING_TYPES entry names no Rust type: {pt}")
    # The reverse direction: every public Cyrius name maps to a Rust item, or is Cyrius-only by a CYRIUS_ONLY rule.
    public = cyrius_public()
    extra_names = sorted(n for n in public if n not in matched)
    rule_hits = {pattern: [] for pattern, _ in CYRIUS_ONLY}
    for n in extra_names:
        rule = next((pattern for pattern, _ in CYRIUS_ONLY if re.fullmatch(pattern, n)), None)
        if rule is None:
            missing.append(f"Cyrius-only name with no CYRIUS_ONLY rule: {n} ({public[n]})")
        else:
            rule_hits[rule].append(n)
    for pattern, hits in rule_hits.items():
        if not hits:
            missing.append(f"CYRIUS_ONLY rule matches no name: {pattern}")
    missing.extend(profile_drift())
    missing.extend(field_store_scan())
    # Every rust-old/src file is accounted for.
    accounted = set(MODULES) | set(PENDING_MODULES) | set(DEFERRED_MODULES) | set(NOT_APPLICABLE_MODULES)
    for path in sorted((root / "rust-old/src").rglob("*.rs")):
        rel = str(path.relative_to(root / "rust-old/src"))[:-3]
        if rel not in accounted:
            missing.append(f"rust-old/src/{rel}.rs is in no module list")
    print(f"surface parity ({', '.join(MODULES)}): {ok} items ok, {derived} of them derived traits "
          f"({len(pending)} pending a later milestone), {len(missing)} missing")
    print(f"reverse: {len(public)} public Cyrius names, {len(public) - len(extra_names)} Rust items and "
          f"{len(extra_names)} Cyrius-only by {len(CYRIUS_ONLY)} rules")
    if "--extra" in sys.argv:
        for pattern, reason in CYRIUS_ONLY:
            print(f"  {pattern} ({len(rule_hits[pattern])}): {', '.join(rule_hits[pattern])}")
    for p in pending:
        print("  PENDING", p)
    for m in missing:
        print("  MISSING", m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
