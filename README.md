# Pravash

> **Pravash** (Sanskrit: प्रवास, journey, flow) is the fluid dynamics library for
> [AGNOS](https://github.com/MacCracken/agnosticos): particle fluids, grid fluids, shallow water, fluid–body
> coupling and the material point method, with deterministic stepping and serializable state.

[![CI](https://github.com/MacCracken/pravash/actions/workflows/ci.yml/badge.svg)](https://github.com/MacCracken/pravash/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

pravash **2.0** is a Cyrius library: a parity port of the Rust crate **1.3.0**, which stays frozen in
[`rust-old/`](rust-old/) as the reference implementation. It is checked against Rust value by value:
- **68,007 golden rows** in 18 modules replay Rust's own results on x86_64 Linux, x86_64 Windows (PE) and aarch64,
  with the same output bits on all three. 67,276 match Rust within their tolerance class. 728 are accepted under an
  ADR instead: 597 that the f64 neighbour keys change, checked against an independent f64 transcription (ADR 0002
  #2), and 131 where glibc misrounds and the port gives the correctly rounded value
  ([ADR 0004](docs/adr/0004-correct-rounding-is-the-accuracy-target.md)). 3 wait on bayan's JSON policies. Every
  grid and SPH solver step row also runs at 2, 3, 4 and 8 threads and gives the one-thread bits.
- **The Rust tests:** all 405 are ported to a test of the same name or accounted for in the
  [Rust-test ledger](docs/development/rust-test-ledger.md): 391 ported, and the 14 tests of the oracle hooks not
  applicable.
- **The surface:** every public Rust item has its Cyrius counterpart, and every public Cyrius name maps back to a
  Rust item or a recorded reason (`scripts/surface_parity.py`), and each of Rust's 157 derives (the serde pair,
  Default, Clone, PartialEq, Debug) has its counterpart.

2.0 changes the API in deliberate places ([ADR 0002](docs/adr/0002-port-api-divergences.md)). Moving from the
Rust crate: [the migration guide](docs/guides/migration-2.0.md). The [port plan](docs/development/pravash-port-plan.md)
records the decisions, and [`state.md`](docs/development/state.md) the current state.

## Features

- **SPH:** the WCSPH solver (symplectic Euler or velocity Verlet), PCISPH, DFSPH and multiphase steps; the poly6, spiky,
  viscosity and Wendland C2 and C4 kernels; the linear and Tait equations of state; CSF surface tension, delta-SPH,
  implicit viscosity, MLS gradient corrections; non-Newtonian (power law, Bingham, Herschel–Bulkley) and Oldroyd-B
  viscoelastic fluids; heat conduction, phase change and combustion through your own reaction model; foam, spray and
  bubbles; particle splitting and merging; z-order sorting and adaptive time steps.
- **Grids:** 2D incompressible Navier–Stokes on a uniform grid (semi-Lagrangian, MacCormack and BFECC advection;
  Gauss–Seidel, multigrid, DST and conjugate-gradient pressure; Smagorinsky; vorticity confinement; buoyancy;
  no-slip, free-slip and periodic walls), a staggered MAC grid with the ghost-fluid method, k-ε turbulence, a
  volume-of-fluid field and an Allen–Cahn phase field.
- **Shallow water:** the non-linear equations with HLL fluxes, well-balanced terrain, wetting and drying, Manning
  friction, wave breaking, Green–Naghdi dispersion and sediment transport.
- **Coupling and MPM:** rigid spheres and boxes in a particle fluid (drag, added mass, two-way forces), a FLIP/PIC
  and APIC solver with a particle level set, and a material point method solver with fluid, neo-Hookean and
  Drucker–Prager materials.
- **The rest of Rust's crate:** buoyancy, drag and flow regimes; point, Lamb–Oseen and Rankine vortices; the bridge
  conversions for pavan, ushma and goonj; a GPU-agnostic compute interface with packed f32 buffers and kernel
  derivatives; the daimon and hoosh endpoint configs; tracing and Chrome-trace profiling through sakshi.
- **Deterministic:** the same inputs give the same bits on every supported target and at every thread count. Each
  transcendental a step calls is pravash's own and correctly rounded.
- **Threads:** a runtime count per solver, `set_threads(n)`, on a persistent pool, in place of the `parallel` feature.
- **JSON:** the 22 types Rust serializes read and write serde_json's documents.
- **Debug:** `x.debug(sb, alternate)` writes Rust's `{:?}` and `{:#?}` text for every type that derives it, the float
  digits included.
- **No aborts:** every fallible call returns a `Result` with a `PV_ERR_*` code, or a neutral value where Rust would
  panic.

## Using it

Declare it in your `cyrius.cyml`; `cyrius deps` vendors it with the stdlib modules it needs:

```toml
[deps.pravash]
git = "https://github.com/MacCracken/pravash.git"
tag = "2.0.0"
modules = ["dist/pravash.cyr"]
```

A block of water stepped 100 times (Rust 1.3.0's Quick Start):

```cyrius
include "lib/pravash.cyr"

fn main(): i64 {
    alloc_init();
    var particles, n = create_particle_block(0.2, 0.5, 0.3, 0.3, 0.02, 0.001);
    var config: PvFluidConfig = PvFluidConfig_water_2d();
    var solver: PvSphSolver = PvSphSolver_new();

    for (var frame = 0; frame < 100; frame = frame + 1) {
        var t, err = solver.step(particles, n, &config, PV_FLUID_MATERIAL_WATER.viscosity);
        if (is_ok(t) == 0) { return 1; }          # err is a PV_ERR_* code
    }

    var ke: f64 = total_kinetic_energy(particles, n);
    fmt_float(ke, 4);
    println("");
    PvSphSolver_drop(&solver);
    fl_free(particles);
    return 0;
}

var r = main();
sys_exit_group(r);
```

`fmt_float` is the stdlib's `fmt`, which the bundle doesn't need, so a program that prints this way lists `fmt` in
its own `[deps] stdlib`.

**The API follows Rust's** (the [migration guide](docs/guides/migration-2.0.md) has the full mapping):
- **Names.** Structs are `Pv<Name>` with methods of the Rust names (`solver.step(..)`, `sw.total_volume()`);
  constructors are `Pv<Name>_<constructor>` (`PvFluidConfig_water_2d`); free functions keep their names; enum
  members and constants carry `PV_`.
- **Results.** A fallible call returns a value-form `Result`: `var t, v = solver.step(..);` then `is_ok(t)`. On
  `Err`, `v` is a `PV_ERR_*` code, and `pv_error_message(v)` gives its text.
- **Slices and collections.** A slice is a pointer and a count; a returned `Vec` is a freelist buffer you free; a
  public `Vec<f64>` field is a `PvColumn`; an `Option` return is an out pointer and 1 / 0.
- **Memory.** What you create, you free: `PvX_drop(&x)` on solvers and fields, `fl_free` on returned buffers. A
  solver's storage is its own, and a step allocates nothing that outlives it.

## Examples

Rust's three examples, ported ([`docs/examples/`](docs/examples/)). Each prints the output in its `.out` file, which
is derived from the oracle.

| Example | What it runs |
|---|---|
| [`basic_fluid`](docs/examples/basic_fluid.cyr) | An SPH dam break, with the kinetic energy and the fastest particle every 10 frames |
| [`shallow_water`](docs/examples/shallow_water.cyr) | A splash in a shallow-water basin with Manning friction; it prints exactly what Rust's prints |
| [`multiphase`](docs/examples/multiphase.cyr) | Oil above water, with interface tension |

## Known limitations

- **SPH trajectories leave Rust's.** The f64 neighbour keys fix Rust's f32 precision loss at large coordinates, but
  near a cell boundary they can order a particle's neighbours differently. The sums then differ in their last bits,
  and a fast or chaotic scene grows that into different numbers: the SPH examples agree with Rust at frame 0 and not
  at frame 10 ([migration guide #7](docs/guides/migration-2.0.md#7-sph-neighbours-use-f64-keys)).
- **Speed:** the median benchmark takes 4.4× Rust 1.3.0's time, from 0.76× (the small SPH solver steps and the DST
  grid steps are level or faster) to 37× (`create_particle_block`, on the freelist's large blocks). At 8 threads the
  SPH step gains 2.2–3.2×, where Rust's rayon gains 1.05–1.95× ([the record](docs/benchmarks-rust-v-cyrius.md)).
  The upstream issues behind the slow rows are filed (port plan §8).

## Where it sits

pravash is a pure simulation library: no rendering, no scene loading. Its intended consumers are
[kiran](https://github.com/MacCracken/kiran) (water, smoke and fire effects),
[joshua](https://github.com/MacCracken/joshua) (fluid simulation) and
[impetus](https://github.com/MacCracken/impetus) (fluid–body interaction); none consumes the Cyrius library yet. Its
math is its own on the step path while hisab's calls allocate unscoped
([ADR 0003](docs/adr/0003-local-stand-ins-for-library-gaps.md)), so a consumer's resolve carries only pravash and a
few stdlib modules.

## Building

Cyrius (the toolchain is pinned in `cyrius.cyml`):

```sh
cyrius deps                                 # vendor the stdlib and hisab (tests only) into lib/
cyrius build src/main.cyr build/pravash     # compile
cyrius test                                 # tests/*.tcyr: the parity harness and the ported Rust tests
cyrius bench                                # tests/*.bcyr
cyrius fuzz                                 # tests/pravash.fcyr: the JSON readers and the PRAVASH_LOG parser
cyrius audit                                # fmt, lint, docs, tests and benches
cyrius distlib                              # dist/pravash.cyr
```

The oracle and the checks against it (`cargo` from the repo root runs the oracle's rustc 1.99.1, pinned by
`rust-toolchain.toml`):

```sh
cargo test --manifest-path rust-old/Cargo.toml --all-features                          # the Rust 1.3.0 oracle
cargo run --release --locked --manifest-path scripts/oracle/Cargo.toml -- --check      # the corpus is current
python3 -I -B scripts/oracle/validate.py .            # the corpus, recomputed without Rust
python3 -I -B scripts/surface_parity.py .             # every Rust item has its Cyrius name, and back
python3 -I -B scripts/rust_test_ledger.py . --check   # every Rust test is accounted for
python3 -I -B scripts/doc_examples.py .               # the docs' code blocks and the examples compile and run
```

## Documentation

- [Migration guide: Rust 1.3 to Cyrius 2.0](docs/guides/migration-2.0.md)
- [Getting started](docs/guides/getting-started.md) (working on pravash)
- [Examples](docs/examples/)
- [CHANGELOG](CHANGELOG.md)
- [Port plan](docs/development/pravash-port-plan.md), [state](docs/development/state.md),
  [roadmap](docs/development/roadmap.md)
- [Port parity](docs/development/port-parity.md), [Rust-test ledger](docs/development/rust-test-ledger.md),
  [benchmarks against Rust](docs/benchmarks-rust-v-cyrius.md)
- [Architecture decisions](docs/adr/)
- [Contributing](CONTRIBUTING.md), [Security policy](SECURITY.md)
- The Rust 1.3.0 [README](rust-old/README.md), [architecture overview](rust-old/docs/architecture/overview.md) and
  [migration guide](rust-old/docs/development/migration.md), kept for reference

## License

GPL-3.0-only. See [LICENSE](LICENSE).
