//! The oracle hooks are read-only.
//!
//! Each solver runs the same scenario twice, once plain and once dumping its
//! full state after every step, and the two final states must be identical.
//! States are compared through their `Debug` text, which writes every f64 and
//! f32 as its shortest round-trip form, so equal text means equal bits for
//! every value except NaN. Each check therefore also refuses a state that
//! holds a NaN. CI runs this file with and without the `parallel` feature.

use std::hint::black_box;

use hisab::{DVec3, SpatialHash, Vec3};
use pravash::coupling::FlipSolver;
use pravash::grid::{BoundaryCondition, FluidGrid, GridConfig, KEpsilon, MacGrid};
use pravash::mpm::{ConstitutiveModel, MpmParticle, MpmSolver};
use pravash::oracle;
use pravash::phase_field::PhaseField;
use pravash::shallow::ShallowWater;
use pravash::sph::{self, MultiPhaseConfig, PhaseProperties, SphSolver};
use pravash::vof::VofField;
use pravash::{FluidConfig, FluidParticle, ParticleArena};

const STEPS: usize = 300;

/// Runs `make` + `steps` × `step` twice and requires the same final
/// fingerprint with and without a fingerprint (which includes the oracle
/// dump) taken after every step.
fn assert_read_only<S>(
    name: &str,
    steps: usize,
    make: impl Fn() -> S,
    step: impl Fn(&mut S, usize),
    fingerprint: impl Fn(&S) -> String,
) {
    let run = |with_dumps: bool| {
        let mut s = make();
        for k in 0..steps {
            step(&mut s, k);
            if with_dumps {
                black_box(fingerprint(&s));
            }
        }
        fingerprint(&s)
    };
    let plain = run(false);
    let dumped = run(true);
    assert!(
        !plain.contains("NaN"),
        "{name}: the state holds a NaN, so Debug text can't prove bit equality"
    );
    assert_eq!(plain, dumped, "{name}: dumping changed the run");
}

// ---------------------------------------------------------------------------
// common
// ---------------------------------------------------------------------------

#[test]
fn arena_dumps_are_read_only() {
    // Every step changes the arena: allocations, frees, and at step 100 one
    // compaction. `compact` makes every handle stale, so the blocks are freed
    // first, in an interleaved order that fragments the free list. Only one
    // compaction, so a difference after it survives to the end of the run.
    let make = || (ParticleArena::new(2048), Vec::new());
    let step = |s: &mut (ParticleArena, Vec<pravash::ArenaHandle>), k: usize| {
        let (arena, handles) = s;
        if k == 100 {
            let all = std::mem::take(handles);
            for h in all.iter().step_by(2).chain(all.iter().skip(1).step_by(2)) {
                arena.free(*h);
            }
            arena.compact();
        } else if k % 5 == 2 && !handles.is_empty() {
            let h = handles.remove((k * 7) % handles.len());
            arena.free(h);
        } else {
            let h = arena
                .alloc(1 + k % 7)
                .expect("the arena never fills in this scenario");
            handles.push(h);
        }
    };
    let fingerprint = |s: &(ParticleArena, Vec<pravash::ArenaHandle>)| {
        let handles: Vec<_> = s.1.iter().map(|h| h.oracle_dump()).collect();
        format!("{:?}|{:?}", s.0.oracle_dump(), handles)
    };
    assert_read_only("ParticleArena", STEPS, make, step, fingerprint);
}

// ---------------------------------------------------------------------------
// shallow
// ---------------------------------------------------------------------------

#[test]
fn shallow_water_dumps_are_read_only() {
    for variant in 0..3 {
        let make = || {
            let mut w = ShallowWater::new(32, 32, 0.1, 1.0).expect("valid grid");
            w.add_disturbance(1.6, 1.6, 0.4, 0.1);
            match variant {
                1 => w.use_riemann = true,
                2 => {
                    w.dispersion_coeff = 0.3;
                    w.use_green_naghdi = true;
                }
                _ => {}
            }
            w
        };
        let step = |w: &mut ShallowWater, _| w.step(0.01).expect("stable step");
        let fingerprint = |w: &ShallowWater| format!("{w:?}|{:?}", w.oracle_dump());
        assert_read_only(
            &format!("ShallowWater variant {variant}"),
            STEPS,
            make,
            step,
            fingerprint,
        );
    }
}

// ---------------------------------------------------------------------------
// grid
// ---------------------------------------------------------------------------

fn smoke_grid(n: usize) -> FluidGrid {
    let mut g = FluidGrid::new(n, n, 1.0).expect("valid grid");
    for y in n / 4..n / 2 {
        for x in n / 3..2 * n / 3 {
            g.density[y * n + x] = 1.0;
            g.vx[y * n + x] = 0.5;
        }
    }
    g
}

fn grid_config(variant: usize) -> GridConfig {
    let mut c = GridConfig::smoke();
    match variant {
        // 0: the default NoSlip boundary takes the DST solve.
        1 => c.boundary = BoundaryCondition::Periodic, // Gauss–Seidel
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
    c
}

#[test]
fn fluid_grid_dumps_are_read_only() {
    for variant in 0..6 {
        let config = grid_config(variant);
        let step = |g: &mut FluidGrid, _| g.step(&config).expect("stable step");
        let fingerprint = |g: &FluidGrid| format!("{g:?}|{:?}", g.oracle_dump());
        assert_read_only(
            &format!("FluidGrid variant {variant}"),
            STEPS,
            || smoke_grid(16),
            step,
            fingerprint,
        );
    }
}

#[test]
fn mac_grid_dumps_are_read_only() {
    let make = || {
        let mut g = MacGrid::new(16, 16, 1.0).expect("valid grid");
        for y in 4..8 {
            for x in 5..11 {
                g.density[y * 16 + x] = 1.0;
            }
        }
        g
    };
    let config = GridConfig::smoke();
    let step = |g: &mut MacGrid, _| g.step(&config).expect("stable step");
    let fingerprint = |g: &MacGrid| format!("{g:?}|{:?}", g.oracle_dump());
    assert_read_only("MacGrid", STEPS, make, step, fingerprint);
}

#[test]
fn k_epsilon_dumps_are_read_only() {
    let n = 16;
    let vx: Vec<f64> = (0..n * n)
        .map(|i| ((i / n) as f64 / n as f64) - 0.5)
        .collect();
    let vy = vec![0.0; n * n];
    let step = |m: &mut KEpsilon, _| m.step(&vx, &vy, n, n, 1.0, 0.01);
    let fingerprint = |m: &KEpsilon| format!("{m:?}|{:?}", m.oracle_dump());
    assert_read_only("KEpsilon", STEPS, || KEpsilon::new(n, n), step, fingerprint);
}

/// A solid-body rotation about the domain centre, at cell centres.
fn rotation(n: usize, dx: f64) -> (Vec<f64>, Vec<f64>) {
    let c = n as f64 * dx * 0.5;
    let mut vx = vec![0.0; n * n];
    let mut vy = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f64 + 0.5) * dx - c;
            let py = (y as f64 + 0.5) * dx - c;
            vx[y * n + x] = -py;
            vy[y * n + x] = px;
        }
    }
    (vx, vy)
}

#[test]
fn vof_dumps_are_read_only() {
    let n = 32;
    let dx = 1.0 / n as f64;
    let (vx, vy) = rotation(n, dx);
    let make = || {
        let mut f = VofField::new(n, n, dx).expect("valid grid");
        f.fill_circle(0.5, 0.3, 0.12);
        f
    };
    let step = |f: &mut VofField, _| f.advect(&vx, &vy, 0.01).expect("valid advection");
    let fingerprint = |f: &VofField| format!("{f:?}|{:?}", f.oracle_dump());
    assert_read_only("VofField", STEPS, make, step, fingerprint);
}

#[test]
fn phase_field_dumps_are_read_only() {
    let n = 32;
    let dx = 1.0 / n as f64;
    let (vx, vy) = rotation(n, dx);
    let make = || {
        let mut f = PhaseField::new(n, n, dx).expect("valid grid");
        f.set_circle(0.5, 0.3, 0.12);
        f
    };
    let step = |f: &mut PhaseField, _| f.step(&vx, &vy, 0.0005).expect("valid step");
    let fingerprint = |f: &PhaseField| format!("{f:?}|{:?}", f.oracle_dump());
    assert_read_only("PhaseField", STEPS, make, step, fingerprint);
}

// ---------------------------------------------------------------------------
// coupling, mpm
// ---------------------------------------------------------------------------

fn particle_block(
    x0: f64,
    y0: f64,
    nx: usize,
    ny: usize,
    spacing: f64,
    mass: f64,
) -> Vec<FluidParticle> {
    let mut v = Vec::with_capacity(nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            v.push(FluidParticle::new_2d(
                x0 + i as f64 * spacing,
                y0 + j as f64 * spacing,
                mass,
            ));
        }
    }
    v
}

#[test]
fn flip_dumps_are_read_only() {
    for use_apic in [false, true] {
        let make = || {
            let mut s = FlipSolver::new(16, 16, 0.1, 0.95).expect("valid grid");
            s.use_apic = use_apic;
            (s, particle_block(0.4, 0.4, 8, 5, 0.05, 0.01))
        };
        let step = |s: &mut (FlipSolver, Vec<FluidParticle>), _| {
            s.0.step(&mut s.1, DVec3::new(0.0, -9.81, 0.0), 0.01)
                .expect("stable step");
        };
        let fingerprint =
            |s: &(FlipSolver, Vec<FluidParticle>)| format!("{:?}|{:?}", s.1, s.0.oracle_dump());
        assert_read_only(
            &format!("FlipSolver apic={use_apic}"),
            STEPS,
            make,
            step,
            fingerprint,
        );
    }
}

#[test]
fn mpm_dumps_are_read_only() {
    let models = [
        ConstitutiveModel::Fluid {
            bulk_modulus: 50.0,
            viscosity: 0.1,
        },
        ConstitutiveModel::NeoHookean {
            youngs_modulus: 1000.0,
            poisson_ratio: 0.3,
        },
        ConstitutiveModel::DruckerPrager {
            youngs_modulus: 1000.0,
            poisson_ratio: 0.3,
            friction_angle: 30.0,
        },
    ];
    for (m, model) in models.iter().enumerate() {
        let make = || {
            let particles = particle_block(0.6, 0.6, 4, 4, 0.05, 0.0025);
            let data = vec![MpmParticle::new(*model, 0.0025); particles.len()];
            (
                MpmSolver::new(16, 16, 0.1).expect("valid grid"),
                particles,
                data,
            )
        };
        let step = |s: &mut (MpmSolver, Vec<FluidParticle>, Vec<MpmParticle>), _| {
            s.0.step(&mut s.1, &mut s.2, DVec3::new(0.0, -9.81, 0.0), 0.001)
                .expect("stable step");
        };
        let fingerprint = |s: &(MpmSolver, Vec<FluidParticle>, Vec<MpmParticle>)| {
            format!("{:?}|{:?}|{:?}", s.1, s.2, s.0.oracle_dump())
        };
        assert_read_only(
            &format!("MpmSolver model {m}"),
            STEPS,
            make,
            step,
            fingerprint,
        );
    }
}

// ---------------------------------------------------------------------------
// sph
// ---------------------------------------------------------------------------

fn sph_config() -> FluidConfig {
    let mut c = FluidConfig::water_2d();
    c.dt = 0.0002;
    c
}

fn sph_block() -> Vec<FluidParticle> {
    // 8 × 8 particles at half the smoothing radius, 2D water mass.
    particle_block(0.3, 0.3, 8, 8, 0.025, 1000.0 * 0.025 * 0.025)
}

type SphState = (SphSolver, Vec<FluidParticle>);

fn sph_fingerprint(s: &SphState) -> String {
    format!("{:?}|{:?}", s.1, s.0.oracle_dump())
}

#[test]
fn sph_dumps_are_read_only() {
    let config = sph_config();
    let two_phase = MultiPhaseConfig::two_phase(
        PhaseProperties {
            rest_density: 1000.0,
            gas_constant: 2000.0,
            viscosity: 0.01,
        },
        PhaseProperties {
            rest_density: 500.0,
            gas_constant: 2000.0,
            viscosity: 0.01,
        },
        0.05,
    );
    for variant in 0..5 {
        let make = || {
            let mut s = SphSolver::new();
            s.use_verlet = variant == 1;
            let mut p = sph_block();
            if variant == 4 {
                for (i, q) in p.iter_mut().enumerate() {
                    q.phase = u8::from(i >= 32);
                }
            }
            (s, p)
        };
        let step = |s: &mut SphState, _| {
            let (solver, particles) = s;
            match variant {
                0 | 1 => solver.step(particles, &config, 0.01),
                2 => solver.step_pcisph(particles, &config, 0.01, 10, 0.01),
                3 => solver.step_dfsph(particles, &config, 0.01, 10, 0.01),
                _ => solver.step_multiphase(particles, &config, &two_phase),
            }
            .expect("stable step");
        };
        assert_read_only(
            &format!("SphSolver variant {variant}"),
            STEPS,
            make,
            step,
            sph_fingerprint,
        );
    }
}

/// The dump's neighbour data is what hisab 1.4.0 holds: each particle's
/// `sph_cell_key` names the cell hisab stores it in, and the CSR lists are
/// well formed and contain the particle itself.
#[test]
fn sph_neighbour_dump_matches_hisab() {
    let config = sph_config();
    let mut solver = SphSolver::new();
    let mut particles = sph_block();
    for _ in 0..20 {
        solver
            .step(&mut particles, &config, 0.01)
            .expect("stable step");
    }
    let d = solver.oracle_dump();
    let h_f32 = config.smoothing_radius as f32;
    assert_eq!(d.cell_size.to_bits(), h_f32.to_bits());
    assert_eq!(d.positions_f32.len(), particles.len());
    assert_eq!(d.grid_len, particles.len());

    let mut hash = SpatialHash::new(d.cell_size).expect("valid cell size");
    for (i, p) in d.positions_f32.iter().enumerate() {
        hash.insert(Vec3::new(p[0], p[1], p[2]), i);
    }
    assert_eq!(hash.cell_count(), d.grid_cells);

    assert_eq!(d.neighbor_offsets.len(), particles.len() + 1);
    assert_eq!(d.neighbor_offsets[0], 0);
    assert_eq!(
        *d.neighbor_offsets.last().expect("n + 1 offsets") as usize,
        d.neighbor_indices.len()
    );
    for (i, p) in d.positions_f32.iter().enumerate() {
        let key = oracle::sph_cell_key(*p, h_f32);
        // The centre of the keyed cell lands in the cell hisab put `i` in.
        let centre = Vec3::new(
            (key[0] as f32 + 0.5) * h_f32,
            (key[1] as f32 + 0.5) * h_f32,
            (key[2] as f32 + 0.5) * h_f32,
        );
        assert!(
            hash.query_cell(centre).contains(&i),
            "particle {i} is not in cell {key:?}"
        );
        let own =
            &d.neighbor_indices[d.neighbor_offsets[i] as usize..d.neighbor_offsets[i + 1] as usize];
        assert!(
            own.contains(&i),
            "particle {i} is missing from its own candidate list"
        );
    }
}

// ---------------------------------------------------------------------------
// hooks
// ---------------------------------------------------------------------------

/// A smooth interior right-hand side on an `n × n` grid with a zero ring.
fn poisson_system(n: usize) -> Vec<f64> {
    let mut div = vec![0.0; n * n];
    for y in 1..n - 1 {
        for x in 1..n - 1 {
            let fx = x as f64 / (n - 1) as f64;
            let fy = y as f64 / (n - 1) as f64;
            div[y * n + x] = (fx * (1.0 - fx)) * (fy * (1.0 - fy));
        }
    }
    div
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn pressure_hooks_are_deterministic_and_agree() {
    let n = 12;
    let div = poisson_system(n);
    let solve = |which: usize| {
        let mut p = vec![0.0; n * n];
        let levels = match which {
            0 => {
                oracle::pressure_solve_gauss_seidel(&mut p, &div, n, n, 4000).expect("GS");
                0
            }
            1 => {
                oracle::pressure_solve_dst(&mut p, &div, n, n).expect("DST");
                0
            }
            2 => {
                oracle::pressure_solve_cg(&mut p, &div, n, n, 400).expect("CG");
                0
            }
            _ => oracle::pressure_solve_multigrid(&mut p, &div, n, n, 4, 20)
                .expect("multigrid")
                .len(),
        };
        (p, levels)
    };
    let dst = solve(1).0;
    let scale = dst.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(scale > 0.0);
    for which in 0..4 {
        let (a, levels) = solve(which);
        let (b, _) = solve(which);
        assert_eq!(bits(&a), bits(&b), "solver {which} is not deterministic");
        assert!(
            a.iter().all(|x| x.is_finite()),
            "solver {which} gave a non-finite value"
        );
        let err = a
            .iter()
            .zip(&dst)
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            err <= 1e-3 * scale,
            "solver {which} is {err} from the DST solution"
        );
        if which == 3 {
            assert!(levels > 0, "multigrid kept no level buffers");
        }
    }
}

#[test]
fn invert_3x3_hook() {
    let inv = oracle::invert_3x3(&[2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0]).expect("regular");
    assert_eq!(
        bits(&inv),
        bits(&[0.5, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.125])
    );
    assert!(oracle::invert_3x3(&[0.0; 9]).is_none());
}

/// The free `sph::step` keeps no state, so it needs no dump; this pins that
/// the oracle build leaves it callable.
#[test]
fn free_sph_step_runs_under_oracle() {
    let config = sph_config();
    let mut particles = sph_block();
    sph::step(&mut particles, &config, 0.01).expect("stable step");
    assert!(particles.iter().all(|p| p.position.x.is_finite()));
}
