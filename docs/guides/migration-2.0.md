# Moving from pravash 1.3 (Rust) to 2.0 (Cyrius)

pravash 2.0 is the Rust crate 1.3.0's functionality, ported to Cyrius, with correct rounding as the accuracy bar
([ADR 0004](../adr/0004-correct-rounding-is-the-accuracy-target.md)). The Rust crate's last release is 1.3.0; its
source is kept in `rust-old/` for reference, and nothing in 2.0 builds, tests or releases from it. This guide is for code that used the Rust crate (kiran, joshua and impetus are the intended
consumers): what to declare, how each Rust idiom reads in Cyrius, and the deliberate changes, each with the code
that handles it. [ADR 0002](../adr/0002-port-api-divergences.md) records every change in full, one row each.

The physics doesn't change. For the same inputs, a 2.0 step computes what Rust 1.3.0 computes, except where 1.3.0 is
inaccurate: where glibc misrounds, 2.0 gives the correctly rounded value, and the SPH neighbour keys are f64 (#7
below); see [What stays the same](#what-stays-the-same).

Every `cyrius` block below compiles and runs against the shipped bundle: `python3 -I -B scripts/doc_examples.py .`
wraps each fragment in a `main`, includes `dist/pravash.cyr`, and fails on a compile warning or a nonzero exit.

## 1. Declare the dependency

```toml
# your cyrius.cyml
[deps.pravash]
git = "https://github.com/MacCracken/pravash.git"
tag = "2.0.0"
modules = ["dist/pravash.cyr"]
```

`cyrius deps` vendors the bundle and the stdlib modules it needs, which are listed in `dist/pravash.deps`. Cargo's
features become one bundle, and dead-code elimination drops what a program doesn't call
([ADR 0002 #5](../adr/0002-port-api-divergences.md)):

| Cargo feature (1.3.0) | 2.0 |
|---|---|
| `sph`, `grid`, `shallow` (the defaults) | in the bundle |
| `buoyancy`, `vortex`, `coupling`, `compute`, `ai`, `logging` | in the bundle |
| `full` | the bundle |
| `parallel` | a thread count you set at run time: `solver.set_threads(n)`, `grid.set_threads(n)` (#6 below) |
| `oracle` | not shipped; it exists only in `rust-old/` to drive the golden generator |

One function is outside the bundle. `KernelDerivatives::autodiff_gradient` records on hisab's tape, and hisab is
only a test-scope dependency of pravash, so `PvKernelDerivatives_autodiff_gradient` is in `src/compute_ad.cyr`. A
consumer that wants it declares hisab itself and lists that file after the bundle (ADR 0002 #26).

## 2. Rust idioms in Cyrius

The names are Rust's. Structs are `Pv<Name>` with methods (`solver.step(..)`), constructors and associated functions
are `Pv<Name>_<name>` (`PvFluidConfig_water_2d`), free functions keep their names (`create_particle_block`,
`total_kinetic_energy`), and enum members and constants carry `PV_`. Two free functions are renamed because one
namespace serves pravash and its consumers: `sph::step` is `sph_step`, and `logging::init` is `logging_init`.

| Rust 1.3.0 | Cyrius 2.0 |
|---|---|
| `SphSolver::new()` | `PvSphSolver_new()`, by value; `PvSphSolver_drop(&s)` releases its buffers |
| `Drop` | explicit: `PvX_drop(&x)` on each solver, field and column-holding type, and `fl_free(p)` on a returned buffer |
| `FluidMaterial::WATER.viscosity` | `PV_FLUID_MATERIAL_WATER.viscosity`; copy the global, never assign to it |
| `DVec3` | `PvVec3`, three f64 (`PvVec3_new(x, y, z)`) |
| `&[FluidParticle]`, `&mut [FluidParticle]` | a pointer and a count: `(ps, n)` |
| `Vec<FluidParticle>` returned | a pointer and a count, in a freelist buffer you `fl_free` |
| a public `Vec<f64>` field | a `PvColumn`: `.len`, `get_f(i)`, `set_f(i, x)`, `push_f(x)`, `fill_f(x)`, `resize(n, 0)` |
| `Option<ArenaHandle>` | an out pointer and 1 / 0: `arena.alloc(n, &h)` |
| `Result<T, PravashError>` | a value-form `Result`: `var t, v = f(..);` then `is_ok(t)`; on `Err`, `v` is a `PV_ERR_*` code |
| `PravashError::InvalidTimestep { dt }` | `PV_ERR_INVALID_TIMESTEP`: the code alone. `pv_error_name(code)` and `pv_error_message(code)` give fixed text |
| a panic: an index out of range, an `expect` | a neutral value: NaN for an f64, 0 for a word, (0, 0) for a view; never an abort |
| an abort when memory runs out | `Err(PV_ERR_OUT_OF_MEMORY)`, or 0 from a constructor or a growth, with nothing changed |
| `NonNewtonianViscosity::PowerLaw { .. }` | a kind and a payload: `PvNonNewtonianModel_power_law(k, n)` (#4 below) |
| `&dyn ReactionProvider` | `dyn PvReactionProvider`, passed as `&provider` |
| `x.clone()` | `PvX_clone(&out, &x)`: 1, or 0 with nothing held; a type with no storage copies by assignment |
| `serde_json::to_string(&x)` | `x.to_json(sb)` onto a `str_builder` |
| `serde_json::from_str::<X>(s)` | `PvX_from_json(buf, len, &out)`, a `Result` whose `Err` is `PV_ERR_DESERIALIZE` |
| `format!("{x:?}")`, `format!("{x:#?}")` | `x.debug(sb, 0)`, `x.debug(sb, 1)` onto a `str_builder`; an enum's `<snake>_debug(k, sb, alternate)` (#13 below) |

A solver step, with its `Result`:

```cyrius
# Rust: let mut ps = create_particle_block([0.2, 0.5], [0.3, 0.3], 0.02, 0.001);
var ps, n = create_particle_block(0.2, 0.5, 0.3, 0.3, 0.02, 0.001);
var config: PvFluidConfig = PvFluidConfig_water_2d();
var solver: PvSphSolver = PvSphSolver_new();

# Rust: solver.step(&mut ps, &config, FluidMaterial::WATER.viscosity)?;
var t, e = solver.step(ps, n, &config, PV_FLUID_MATERIAL_WATER.viscosity);
if (is_ok(t) == 0) { return 1; }
var ke: f64 = total_kinetic_energy(ps, n);

# Rust: Err(PravashError::InvalidTimestep { dt: -1.0 }). Here the code alone: "InvalidTimestep".
var bad: PvFluidConfig = PvFluidConfig_water_2d();
bad.dt = -1.0;
var t2, e2 = solver.step(ps, n, &bad, 0.001);
if (is_ok(t2) == 1 || e2 != PV_ERR_INVALID_TIMESTEP) { return 1; }
if (streq(pv_error_name(e2), "InvalidTimestep") == 0) { return 1; }

PvSphSolver_drop(&solver);
fl_free(ps);
if (ke <= 0.0) { return 1; }
```

A solver's public fields are columns, and a missing index gives a neutral value where Rust panics:

```cyrius
# Rust: let mut sw = ShallowWater::new(40, 40, 0.1, 1.0)?;
var sw: PvShallowWater;
var t, e = PvShallowWater_new(40, 40, 0.1, 1.0, &sw);
if (is_ok(t) == 0) { return 1; }
sw.manning_n.fill_f(0.025);                     # Rust: sw.manning_n.fill(0.025);
var ok = sw.ground.set_f(41, 0.2);              # Rust: sw.ground[41] = 0.2;  0 out of range
var z: f64 = sw.ground.get_f(41);               # Rust: sw.ground[41];  NaN out of range
var cells = sw.height.len;                      # Rust: sw.height.len()
var far: f64 = sw.depth_at(400, 400);           # NaN, where Rust panics

var copy: PvShallowWater;
if (PvShallowWater_clone(&copy, &sw) == 0) { return 1; }    # Rust: let copy = sw.clone();
PvShallowWater_drop(&copy);
PvShallowWater_drop(&sw);
if (ok != 1 || z != 0.2 || cells != 1600 || far == far) { return 1; }    # far == far is false for NaN
```

The arena's `Option` and its slice views:

```cyrius
var arena: *PvParticleArena = PvParticleArena_new(64);         # Rust: ParticleArena::new(64)
if (arena == 0) { return 1; }
var h: PvArenaHandle;
if (arena.alloc(10, &h) == 0) { return 1; }                   # Rust: let h = arena.alloc(10)?;  (None: 0)
var ps, n = arena.get(h);                                      # Rust: arena.get(h): &[FluidParticle]
arena.free(h);
var stale, none = arena.get(h);                                # (0, 0) for a freed handle
PvParticleArena_drop(arena);
if (n != 10 || none != 0) { return 1; }
```

## 3. The changes

### #1 Errors are codes, and nothing aborts

`PravashError`'s variants are the codes `PV_ERR_INVALID_PARTICLE_COUNT` (1) through `PV_ERR_INVALID_PARAMETER` (9),
in Rust's order, with `PV_ERR_NONE` (0) for Ok. Their detail strings (`Diverged { reason }`, thiserror's text) are
gone. Two codes are new: `PV_ERR_DESERIALIZE` (10), for a JSON document the port refuses, and
`PV_ERR_OUT_OF_MEMORY` (11), where Rust aborts. A call that runs out of memory returns that code before it changes
anything, unless its comment says otherwise. Enum values are append-only: a code keeps its number in every 2.x
release (ADR 0002 #1, #8, #11).

### #2 Memory is yours to release

Rust frees on drop. In 2.0 each type that holds storage has a `PvX_drop`, and each returned buffer is a freelist
block you pass to `fl_free`. A drop leaves the value empty and usable again (ADR 0002 #9). Each solver owns its
buffers and reuses them across steps, so a step allocates nothing that outlives it once the first step has grown
them.

### #3 Slices, lists and returned vectors

A slice is a pointer and a count, and the CSR neighbour lists the physics extensions take are two of them, of
8-byte words. `SphSolver::neighbor_lists()` returns a `PvNeighborLists` with all four (ADR 0002 #18, #19):

```cyrius
var ps, n = create_particle_block(0.2, 0.3, 0.2, 0.2, 0.02, 0.001);
var config: PvFluidConfig = PvFluidConfig_water_2d();
var solver: PvSphSolver = PvSphSolver_new();
var t, e = solver.step(ps, n, &config, 0.001);

# Rust: let (offsets, indices) = solver.neighbor_lists();
#       update_heat(&mut ps, offsets, indices, h, &HeatConfig::new(1.4e-7), dt);
var lists: PvNeighborLists = solver.neighbor_lists();
var heat: PvHeatConfig = PvHeatConfig_new(0.00000014);
var ok = update_heat(ps, n, lists.offsets, lists.n_off, lists.indices, lists.n_idx, config.smoothing_radius,
    &heat, config.dt);
PvSphSolver_drop(&solver);
fl_free(ps);
if (is_ok(t) == 0 || ok != 1) { return 1; }
```

`update_heat`, `update_viscoelastic`, `apply_delta_sph` and `apply_implicit_viscosity` return 1, or 0 with nothing
changed where Rust warns and returns on invalid lists.

### #4 Data-carrying enums are a kind and a payload

`NonNewtonianViscosity`, `BodyShape` and `ConstitutiveModel` carry fields per variant. In 2.0 each is an enum of kinds
(`PvNonNewtonianViscosity`, `PvBodyShape`, `PvConstitutiveModel`) and a payload struct holding every variant's fields
(`PvNonNewtonianModel`, `PvShape`, `PvConstitutiveLaw`), with a constructor per variant. A `match` becomes a test of
`kind` (ADR 0002 #15, #23, #25):

```cyrius
# Rust: let m = NonNewtonianViscosity::PowerLaw { consistency: 0.5, power_index: 0.8 };
var m: PvNonNewtonianModel = PvNonNewtonianModel_power_law(0.5, 0.8);
# Rust: m.effective_viscosity(10.0)
var mu: f64 = non_newtonian_viscosity_effective_viscosity(&m, 10.0);

# Rust: match shape { BodyShape::Sphere { radius } => radius, BodyShape::Box { .. } => .. }
var shape: PvShape = PvShape_sphere(0.25);
var r: f64 = 0.0;
if (shape.kind == PV_BODY_SHAPE_SPHERE) { r = shape.radius; }
var body: PvRigidBody = PvRigidBody_new(PvVec3_new(0.5, 0.5, 0.0), 2.0, shape);
if (m.kind != PV_NON_NEWTONIAN_VISCOSITY_POWER_LAW || mu <= 0.0 || r != 0.25) { return 1; }
```

### #5 Traits through `dyn`

`ReactionProvider`, `ComputeBackend` and `NeuralCorrector` are `PvReactionProvider`, `PvComputeBackend` and
`PvNeuralCorrector`. A call through a `dyn` returns one value, so `react`'s `(f64, f64)` is a `PvReaction { consumed,
heat }`, and the compute traits return a status word, `PV_ERR_NONE` or a code (ADR 0002 #18, #26):

```cyrius
include "lib/pravash.cyr"

# Rust: struct Burn { .. }  impl ReactionProvider for Burn { fn react(..) -> (f64, f64) { .. } }
struct Burn {
    ignition: f64;
    rate: f64;
    heat_per_fuel: f64;
}

impl PvReactionProvider for Burn {
    # Above the ignition temperature, fuel burns at a fixed rate and gives its heat.
    fn react(self, temperature: f64, fuel: f64, dt: f64): PvReaction {
        var r: PvReaction;
        r.consumed = 0.0;
        r.heat = 0.0;
        if (temperature > self.ignition) {
            var burn: f64 = self.rate * dt;
            r.consumed = burn;
            if (burn > fuel) { r.consumed = fuel; }
            r.heat = r.consumed * self.heat_per_fuel;
        }
        return r;
    }
}

fn main(): i64 {
    alloc_init();
    var burn: Burn;
    burn.ignition = 600.0;
    burn.rate = 50.0;
    burn.heat_per_fuel = 2000.0;
    var p: PvFluidParticle = PvFluidParticle_new_2d(0.0, 0.0, 1.0);
    p.fuel = 1.0;
    p.temperature = 900.0;
    update_reaction(&p, 1, &burn, 0.001);       # Rust: update_reaction(&mut ps, &burn, 0.001);
    if (p.fuel != 0.95 || p.temperature <= 900.0) { return 1; }
    return 0;
}
```

### #6 Threads are a count, not a feature

Rust 1.3.0 ran three SPH loops and three grid loops on rayon when built with `parallel`. 2.0 has no build switch:
`set_threads(n)` on a `PvSphSolver` or a `PvFluidGrid` starts a persistent pool at the next step, and every count
gives the bits of one thread (ADR 0002 #4). `step_pcisph`, `step_dfsph` and the other grids have no parallel loops in
Rust and run serially at any count.

```cyrius
var a, n = create_particle_block(0.2, 0.3, 0.2, 0.2, 0.02, 0.001);
var b, m = create_particle_block(0.2, 0.3, 0.2, 0.2, 0.02, 0.001);
var config: PvFluidConfig = PvFluidConfig_water_2d();
var one: PvSphSolver = PvSphSolver_new();
var four: PvSphSolver = PvSphSolver_new();
var set = four.set_threads(4);
for (var k = 0; k < 10; k = k + 1) {
    var t1, e1 = one.step(a, n, &config, 0.001);
    var t4, e4 = four.step(b, m, &config, 0.001);
}
var same = memeq(a, b, n * sizeof(PvFluidParticle));
PvSphSolver_drop(&one);
PvSphSolver_drop(&four);
fl_free(a);
fl_free(b);
if (same != 1) { return 1; }                   # four threads, the serial bits
```

### #7 SPH neighbours use f64 keys

Rust 1.3.0 keyed its SPH and coupling neighbour grids in f32 (hisab 1.4.0's `SpatialHash`). That loses precision at
large coordinates: a probe can land in the wrong cell and list a cell twice. 2.0 keys each point by
floor(x · fl(1/h)) in f64, and a particle's list is the 27 cells around its own key, each cell's points in index
order (ADR 0002 #2). The neighbour sets are the same wherever Rust's were right, but the order of a particle's list
can differ near a cell boundary, so the sums run in another order, and a trajectory can leave Rust's in its last
bits from that step on.

The ported examples show both sides ([`docs/examples/`](../examples/)). The shallow-water example prints exactly what
Rust's prints. The two SPH examples match Rust's first frame, then separate. In `basic_fluid`, at the second step, the
f32 keys put a particle at x = 0.3999999999999997 into the next cell, so 49 particles sum the same neighbours in another
order, and the scene, which reaches 6,800 m/s by frame 10, grows the difference. validate.py's transcription of the
solver reproduces Rust's numbers on f32 keys and the port's on f64 keys, so the keys are the only difference
(`scripts/oracle/examples.py`).

### #8 `sort_by_zorder` breaks ties by index

Rust's unstable sort orders particles with equal z-order keys in a way only core's sort could reproduce. 2.0's sort
is stable: equal keys keep their input order. It returns 1 sorted, or 0 with nothing changed (fewer than two
particles, a cell size not finite and above 0, or no memory for its scratch) (ADR 0002 #3, #16).

### #9 JSON

The 22 serde types read and write serde_json's documents: every document serde_json writes from Rust 1.3.0 decodes to
the same bits, and every 2.0 document reads back exactly. An f64 is printed with bayan's Grisu2 digits, which round
trip but aren't always serde_json's shortest form, so the text can differ in its digits (ADR 0002 #6, #10). Three
documents serde_json reads are refused or read otherwise: an integer above `i64::MAX`, a repeated unknown field, and
the integer `-0`, read as +0.

```cyrius
var config: PvFluidConfig = PvFluidConfig_water_2d();
var sb = str_builder_new();
if (config.to_json(sb) != 0) { return 1; }                     # Rust: serde_json::to_string(&config)?
var text = str_builder_build(sb);
var back: PvFluidConfig;
var t, e = PvFluidConfig_from_json(str_data(text), str_len(text), &back);    # Rust: serde_json::from_str(&text)?
if (is_ok(t) == 0 || back.smoothing_radius != config.smoothing_radius) { return 1; }

var broken = "{\"dt\": }";
var t2, e2 = PvFluidConfig_from_json(broken, strlen(broken), &back);
if (is_ok(t2) == 1 || e2 != PV_ERR_DESERIALIZE) { return 1; }
```

### #10 Logging is sakshi

The crate's 60 `tracing` spans and 12 warnings go through sakshi, the AGNOS logging library, with Rust's names and
fields. `logging::init()`, `init_with_level(level)` and `init_profiling()` are `logging_init()`,
`logging_init_with_level(level, len)` and `logging_init_profiling(&guard)`; `PRAVASH_LOG` takes Rust's filter syntax,
and the profiling guard writes tracing-chrome's trace file. One difference shows without any setup: Rust with no
subscriber printed nothing, while sakshi's default prints warnings and errors to stderr, as every library on sakshi
does. To keep pravash quiet, raise sakshi's level before the first call (ADR 0002 #28):

```cyrius
var quiet = sakshi_set_level(SK_ERROR);         # pravash's warnings no longer print
```

### #11 The ai configs hold bytes

`DaimonConfig` and `HooshConfig` keep each string as a pointer and a length, with an `api_key` of 0 for `None`. You
set the strings with methods that copy them, since the structs are `#[non_exhaustive]` and couldn't be built by a
literal outside the crate anyway. Their `Debug` is `c.debug(sb, alternate)`, Rust's `{:?}` and `{:#?}` text, and
still redacts the key (ADR 0002 #27):

```cyrius
var d: PvDaimonConfig = PvDaimonConfig_default();                # endpoint "http://localhost:8090", no key
if (d.set_api_key("sk-test", 7) == 0) { return 1; }              # Rust: d.api_key = Some("sk-test".into());
var sb = str_builder_new();
var rc = d.debug(sb, 0);                                        # Rust: format!("{d:?}")
var text = str_builder_build(sb);
PvDaimonConfig_drop(&d);
if (rc != 0 || str_len(text) == 0) { return 1; }                # ... api_key: Some("<redacted>") }
```

### #12 Results are the same on every target

This one needs no code change. Rust 1.3.0 promised the same bits only on one platform. 2.0 gives the same bits on
x86_64 Linux, x86_64 Windows and aarch64, because every transcendental a step calls is pravash's own and correctly
rounded (ADR 0002 #7).

### #13 Debug writes onto a builder

Rust's derived `Debug` is `x.debug(sb, alternate)` on each struct: Rust's `{:?}` text with alternate 0 and `{:#?}`
with 1, onto a `str_builder`, returning 0, or -1 when the builder can't grow. A unit-variant enum's kind word prints
through `<snake>_debug(k, sb, alternate)` (`flow_regime_debug`), a payload enum's through
`<snake>_debug(&payload, sb, alternate)` (`body_shape_debug`). The text is Rust's to the byte: the field order, the
private scratch, the float digits (ADR 0002 #29).

```cyrius
var c: PvFluidConfig = PvFluidConfig_water_2d();
var sb = str_builder_new();
if (c.debug(sb, 0) != 0) { return 1; }                          # Rust: format!("{c:?}")
var text = str_builder_build(sb);
var head = "FluidConfig { dt: 0.001, gravity: DVec3(0.0, -9.81, 0.0), smoothing_radius: 0.05, ";
if (memeq(str_data(text), head, strlen(head)) == 0) { return 1; }
var regime = str_builder_new();
if (flow_regime_debug(PV_FLOW_REGIME_LAMINAR, regime, 1) != 0) { return 1; }    # Rust: format!("{r:#?}")
```

## What stays the same

- **The physics.** The golden corpus replays Rust 1.3.0's own results for every module: single calls, single steps
  and multi-step trajectories of each solver. Two kinds of rows are accepted under an ADR instead of matching
  Rust's bits ([port-parity.md](../development/port-parity.md)): those the f64 neighbour keys change (#7), checked
  against validate.py's f64 transcription instead, and those where glibc misrounds one of `exp`, `ln`, `log10`,
  `powf`, `sin`, `cos`, `atan2` or `hypot`, checked against an MPFR-corrected oracle
  ([ADR 0004](../adr/0004-correct-rounding-is-the-accuracy-target.md)). Rust's std calls glibc, and 2.0 computes the
  correctly rounded value, so a trajectory can drift from Rust's from that step on.
- **Defaults and constants** are Rust's, generated from the oracle.
- **Validation**: each config check and each refusal is Rust's, with the same error variant as a code.
- **The tests**: all 405 Rust tests are ported or accounted for in the
  [Rust-test ledger](../development/rust-test-ledger.md).

## Not carried over

- The `oracle` feature (port tooling, never public API).
- `PravashError`'s detail strings and thiserror's messages (#1), and with them its `Debug` text:
  `pv_error_name(code)` gives the variant's name.

## Known limitations

- **Speed.** The median benchmark takes 4.4× Rust's time, from 0.76× (the small SPH solver steps and the DST grid
  steps are level with Rust or faster) to 37× (`create_particle_block`, on the freelist's large blocks). With 8
  threads the SPH step gains 2.2–3.2× where Rust's rayon gains 1.05–1.95×. The upstream issues behind the slow rows
  are filed ([the record](../benchmarks-rust-v-cyrius.md)).
- **Tracing is single-threaded.** sakshi's span stack belongs to the process, so the worker pool emits nothing, and a
  span past a 16-deep stack isn't recorded.
