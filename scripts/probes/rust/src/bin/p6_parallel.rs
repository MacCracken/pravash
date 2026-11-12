//! Probe 6, the Rust side (port plan §4 step 9.6): does the oracle's `parallel` build (rayon) give
//! the serial build's bits?
//!
//! Every solver runs the scenarios `rust-old/tests/oracle.rs` uses (300 steps each), and prints one
//! FNV-1a digest of its final state: the `Debug` text of the public state and the oracle dump, which
//! is exact for every non-NaN value (the harness refuses a NaN). Built twice and compared:
//!
//!   cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin p6_parallel > build/p6_serial.txt
//!   cargo run --release --locked --manifest-path scripts/probes/rust/Cargo.toml --bin p6_parallel --features parallel > build/p6_parallel.txt
//!   diff build/p6_serial.txt build/p6_parallel.txt

use hisab::DVec3;
use pravash::coupling::FlipSolver;
use pravash::grid::{BoundaryCondition, FluidGrid, GridConfig, MacGrid};
use pravash::shallow::ShallowWater;
use pravash::sph::{MultiPhaseConfig, PhaseProperties, SphSolver};
use pravash::{FluidConfig, FluidParticle};

const STEPS: usize = 300;

fn fnv(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn report(name: &str, text: &str) {
    assert!(!text.contains("NaN"), "{name}: NaN in the state");
    println!("{name}\t{:#018x}", fnv(text));
}

fn block(x0: f64, y0: f64, nx: usize, ny: usize, spacing: f64, mass: f64) -> Vec<FluidParticle> {
    let mut v = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            v.push(FluidParticle::new_2d(x0 + i as f64 * spacing, y0 + j as f64 * spacing, mass));
        }
    }
    v
}

fn main() {
    println!("# build: {}", if cfg!(feature = "parallel") { "parallel (rayon)" } else { "serial" });

    // SPH: the five step variants.
    let mut cfg = FluidConfig::water_2d();
    cfg.dt = 0.0002;
    let two_phase = MultiPhaseConfig::two_phase(
        PhaseProperties { rest_density: 1000.0, gas_constant: 2000.0, viscosity: 0.01 },
        PhaseProperties { rest_density: 500.0, gas_constant: 2000.0, viscosity: 0.01 },
        0.05,
    );
    for variant in 0..5 {
        let mut s = SphSolver::new();
        s.use_verlet = variant == 1;
        let mut ps = block(0.3, 0.3, 8, 8, 0.025, 0.625);
        if variant == 4 {
            for (i, q) in ps.iter_mut().enumerate() {
                q.phase = u8::from(i >= 32);
            }
        }
        for _ in 0..STEPS {
            match variant {
                0 | 1 => s.step(&mut ps, &cfg, 0.01),
                2 => s.step_pcisph(&mut ps, &cfg, 0.01, 10, 0.01),
                3 => s.step_dfsph(&mut ps, &cfg, 0.01, 10, 0.01),
                _ => s.step_multiphase(&mut ps, &cfg, &two_phase),
            }
            .expect("stable step");
        }
        report(&format!("SphSolver variant {variant}"), &format!("{ps:?}|{:?}", s.oracle_dump()));
    }

    // FluidGrid: every pressure solver, MacCormack, BFECC.
    for variant in 0..6 {
        let mut c = GridConfig::smoke();
        match variant {
            1 => c.boundary = BoundaryCondition::Periodic,
            2 => c.use_cg = true,
            3 => c.use_multigrid = true,
            4 => {
                c.use_maccormack = true;
                c.smagorinsky_cs = 0.1;
                c.boundary = BoundaryCondition::FreeSlip;
            }
            5 => c.use_bfecc = true,
            _ => {}
        }
        let n = 16;
        let mut g = FluidGrid::new(n, n, 1.0).expect("grid");
        for y in n / 4..n / 2 {
            for x in n / 3..2 * n / 3 {
                g.density[y * n + x] = 1.0;
                g.vx[y * n + x] = 0.5;
            }
        }
        for _ in 0..STEPS {
            g.step(&c).expect("stable step");
        }
        report(&format!("FluidGrid variant {variant}"), &format!("{g:?}|{:?}", g.oracle_dump()));
    }

    let mut mg = MacGrid::new(16, 16, 1.0).expect("grid");
    for y in 4..8 {
        for x in 5..11 {
            mg.density[y * 16 + x] = 1.0;
        }
    }
    for _ in 0..STEPS {
        mg.step(&GridConfig::smoke()).expect("stable step");
    }
    report("MacGrid", &format!("{mg:?}|{:?}", mg.oracle_dump()));

    let mut w = ShallowWater::new(32, 32, 0.1, 1.0).expect("grid");
    w.add_disturbance(1.6, 1.6, 0.4, 0.1);
    for _ in 0..STEPS {
        w.step(0.01).expect("stable step");
    }
    report("ShallowWater", &format!("{w:?}|{:?}", w.oracle_dump()));

    let mut f = FlipSolver::new(16, 16, 0.1, 0.95).expect("grid");
    f.use_apic = true;
    let mut fp = block(0.4, 0.4, 8, 5, 0.05, 0.01);
    for _ in 0..STEPS {
        f.step(&mut fp, DVec3::new(0.0, -9.81, 0.0), 0.01).expect("stable step");
    }
    report("FlipSolver (APIC)", &format!("{fp:?}|{:?}", f.oracle_dump()));
}
