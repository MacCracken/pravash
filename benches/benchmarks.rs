use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hisab::DVec3;
use pravash::common::{FluidConfig, FluidMaterial, FluidParticle};
use pravash::coupling::{self, BodyShape, FlipSolver, RigidBody};
use pravash::grid::{FluidGrid, GridConfig};
use pravash::shallow::ShallowWater;
use pravash::sph::{self, SphSolver};

// ── SPH Benchmarks ──────────────────────────────────────────────────────────

fn bench_sph_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("sph_kernels");
    group.bench_function("poly6", |b| {
        b.iter(|| sph::kernel_poly6(black_box(0.3), black_box(1.0)))
    });
    group.bench_function("spiky_grad", |b| {
        b.iter(|| sph::kernel_spiky_grad(black_box(0.3), black_box(1.0)))
    });
    group.bench_function("viscosity_laplacian", |b| {
        b.iter(|| sph::kernel_viscosity_laplacian(black_box(0.3), black_box(1.0)))
    });
    group.finish();
}

fn bench_sph_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("sph_density");
    for &n in &[100, 500, 1000] {
        let particles =
            sph::create_particle_block([0.0, 0.0], [1.0, 1.0], 1.0 / (n as f64).sqrt(), 0.001);
        let h = 0.05;
        group.bench_function(format!("{n}_particles"), |b| {
            b.iter(|| sph::compute_density(black_box(0), black_box(&particles), black_box(h)))
        });
    }
    group.finish();
}

fn bench_sph_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("sph_step");
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    for &n in &[25, 100, 225] {
        let spacing = 1.0 / (n as f64).sqrt();
        let particles = sph::create_particle_block([0.1, 0.3], [0.5, 0.5], spacing, 0.001);
        group.bench_function(format!("{}_particles", particles.len()), |b| {
            b.iter_batched(
                || particles.clone(),
                |mut p| sph::step(&mut p, &config, viscosity).unwrap(),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_sph_pressure_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("sph_pressure_force");
    let mut particles = sph::create_particle_block([0.0, 0.0], [0.5, 0.5], 0.02, 0.001);
    // Give particles realistic density/pressure
    for p in particles.iter_mut() {
        p.density = 1000.0;
        p.pressure = 100.0;
    }
    let h = 0.05;
    group.bench_function(format!("{}_particles", particles.len()), |b| {
        b.iter(|| sph::pressure_force(black_box(0), black_box(&particles), black_box(h)))
    });
    group.finish();
}

// ── Grid Benchmarks ─────────────────────────────────────────────────────────

fn bench_grid_diffuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_diffuse");
    for &n in &[32, 64, 128] {
        let mut field = vec![0.0; n * n];
        field[n * n / 2] = 100.0;
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter_batched(
                || field.clone(),
                |mut f| FluidGrid::diffuse(&mut f, n, n, 0.1, 0.01, 0.1, 20),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_grid_max_speed(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_max_speed");
    for &n in &[32, 64, 128] {
        let mut grid = FluidGrid::new(n, n, 0.01).unwrap();
        for i in 0..grid.vx.len() {
            grid.vx[i] = (i as f64 * 0.01).sin();
            grid.vy[i] = (i as f64 * 0.01).cos();
        }
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter(|| black_box(&grid).max_speed())
        });
    }
    group.finish();
}

fn bench_grid_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_step");
    let config = GridConfig::smoke();
    for &n in &[15, 30, 63] {
        let mut grid = FluidGrid::new(n, n, 0.1).unwrap();
        // Add some density and velocity for non-trivial work
        let mid = n / 2;
        for x in mid - 2..mid + 2 {
            let i = 2 * n + x;
            grid.density[i] = 1.0;
            grid.vy[i] = 1.0;
        }
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter_batched(
                || grid.clone(),
                |mut g| g.step(&config).unwrap(),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_grid_step_maccormack(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_step_maccormack");
    let mut config = GridConfig::smoke();
    config.use_maccormack = true;
    config.dt = 0.01;
    for &n in &[15, 30, 63] {
        let mut grid = FluidGrid::new(n, n, 0.1).unwrap();
        let mid = n / 2;
        for x in mid - 2..mid + 2 {
            let i = 2 * n + x;
            grid.density[i] = 1.0;
            grid.vy[i] = 1.0;
        }
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter_batched(
                || grid.clone(),
                |mut g| g.step(&config).unwrap(),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_grid_step_periodic(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_step_periodic");
    let mut config = GridConfig::smoke();
    config.boundary = pravash::grid::BoundaryCondition::Periodic;
    for &n in &[15, 30, 63] {
        let mut grid = FluidGrid::new(n, n, 0.1).unwrap();
        let mid = n / 2;
        for x in mid - 2..mid + 2 {
            let i = 2 * n + x;
            grid.density[i] = 1.0;
            grid.vy[i] = 1.0;
        }
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter_batched(
                || grid.clone(),
                |mut g| g.step(&config).unwrap(),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

// ── Shallow Water Benchmarks ────────────────────────────────────────────────

fn bench_shallow_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("shallow_step");
    for &n in &[32, 64, 128] {
        let mut sw = ShallowWater::new(n, n, 0.1, 1.0).unwrap();
        sw.add_disturbance(n as f64 * 0.1 / 2.0, n as f64 * 0.1 / 2.0, 0.5, 0.3);
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter_batched(
                || sw.clone(),
                |mut s| s.step(0.001).unwrap(),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_shallow_volume(c: &mut Criterion) {
    let mut group = c.benchmark_group("shallow_volume");
    for &n in &[32, 64, 128] {
        let sw = ShallowWater::new(n, n, 0.1, 1.0).unwrap();
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter(|| black_box(&sw).total_volume())
        });
    }
    group.finish();
}

// ── SphSolver Benchmarks (spatial hash) ─────────────────────────────────────

fn bench_solver_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_step");
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    for &n in &[25, 100, 225, 625, 1000] {
        let spacing = 1.0 / (n as f64).sqrt();
        let particles = sph::create_particle_block([0.1, 0.3], [0.5, 0.5], spacing, 0.001);
        let mut solver = SphSolver::new();
        group.bench_function(format!("{}_particles", particles.len()), |b| {
            b.iter_batched(
                || particles.clone(),
                |mut p| solver.step(&mut p, &config, viscosity).unwrap(),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_solver_step_surface_tension(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_step_st");
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    for &n in &[100, 225, 625] {
        let spacing = 1.0 / (n as f64).sqrt();
        let particles = sph::create_particle_block([0.1, 0.3], [0.5, 0.5], spacing, 0.001);
        let mut solver = SphSolver::with_surface_tension(0.072);
        group.bench_function(format!("{}_particles", particles.len()), |b| {
            b.iter_batched(
                || particles.clone(),
                |mut p| solver.step(&mut p, &config, viscosity).unwrap(),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_pcisph_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcisph_step");
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    for &n in &[25, 100, 225] {
        let spacing = 1.0 / (n as f64).sqrt();
        let particles = sph::create_particle_block([0.1, 0.3], [0.5, 0.5], spacing, 0.001);
        let mut solver = SphSolver::new();
        group.bench_function(format!("{}_particles", particles.len()), |b| {
            b.iter_batched(
                || particles.clone(),
                |mut p| {
                    solver
                        .step_pcisph(&mut p, &config, viscosity, 5, 0.01)
                        .unwrap()
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_sph_viscosity_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("sph_viscosity_force");
    let mut particles = sph::create_particle_block([0.0, 0.0], [0.5, 0.5], 0.02, 0.001);
    for (i, p) in particles.iter_mut().enumerate() {
        p.density = 1000.0;
        p.velocity = DVec3::new((i as f64 * 0.1).sin(), (i as f64 * 0.1).cos(), 0.0);
    }
    let h = 0.05;
    group.bench_function(format!("{}_particles", particles.len()), |b| {
        b.iter(|| {
            sph::viscosity_force(
                black_box(0),
                black_box(&particles),
                black_box(h),
                black_box(0.001),
            )
        })
    });
    group.finish();
}

// ── Particle Creation ───────────────────────────────────────────────────────

fn bench_create_particle_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_particle_block");
    for &n in &[100, 500, 1000] {
        let spacing = 1.0 / (n as f64).sqrt();
        group.bench_function(format!("{n}_particles"), |b| {
            b.iter(|| {
                sph::create_particle_block(
                    black_box([0.0, 0.0]),
                    black_box([1.0, 1.0]),
                    black_box(spacing),
                    black_box(0.001),
                )
            })
        });
    }
    group.finish();
}

// ── Coupling Benchmarks ─────────────────────────────────────────────────────

fn bench_coupling_sph_bodies(c: &mut Criterion) {
    let mut group = c.benchmark_group("coupling_sph_bodies");
    let particles_base = sph::create_particle_block([0.1, 0.1], [0.5, 0.5], 0.02, 0.001);
    let bodies = vec![
        RigidBody::new(
            DVec3::new(0.3, 0.3, 0.0),
            1.0,
            BodyShape::Sphere { radius: 0.1 },
        ),
        RigidBody::new(
            DVec3::new(0.5, 0.3, 0.0),
            1.0,
            BodyShape::Box {
                half_extents: [0.05, 0.05, 0.05],
            },
        ),
    ];
    group.bench_function(format!("{}_particles", particles_base.len()), |b| {
        b.iter_batched(
            || (particles_base.clone(), bodies.clone()),
            |(mut p, mut bo)| coupling::couple_sph_bodies(&mut p, &mut bo, 0.05, 1000.0, 10.0),
            criterion::BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_particle_level_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_level_set");
    for &(grid_n, n_particles) in &[(32, 100), (64, 500), (32, 1000)] {
        let particles = sph::create_particle_block(
            [0.1, 0.1],
            [0.5, 0.5],
            0.5 / (n_particles as f64).sqrt(),
            0.001,
        );
        let mut level_set = vec![0.0; grid_n * grid_n];
        group.bench_function(
            format!("{grid_n}x{grid_n}_{}_particles", particles.len()),
            |b| {
                b.iter(|| {
                    coupling::particle_level_set(
                        black_box(&mut level_set),
                        black_box(grid_n),
                        black_box(grid_n),
                        black_box(0.1),
                        black_box(&particles),
                        black_box(0.05),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_drag_from_particles(c: &mut Criterion) {
    let mut group = c.benchmark_group("drag_from_particles");
    let particles = sph::create_particle_block([0.1, 0.1], [0.5, 0.5], 0.02, 0.001);
    let mut body = RigidBody::new(
        DVec3::new(0.3, 0.3, 0.0),
        1.0,
        BodyShape::Sphere { radius: 0.1 },
    );
    body.velocity = DVec3::new(1.0, 0.0, 0.0);
    group.bench_function(format!("{}_particles", particles.len()), |b| {
        b.iter(|| {
            coupling::drag_from_particles(
                black_box(&body),
                black_box(&particles),
                black_box(1000.0),
                black_box(0.47),
                black_box(0.15),
            )
        })
    });
    group.finish();
}

fn bench_shallow_disturbance(c: &mut Criterion) {
    let mut group = c.benchmark_group("shallow_disturbance");
    for &n in &[32, 64, 128] {
        let sw = ShallowWater::new(n, n, 0.1, 1.0).unwrap();
        let cx = n as f64 * 0.1 / 2.0;
        group.bench_function(format!("{n}x{n}"), |b| {
            b.iter_batched(
                || sw.clone(),
                |mut s| s.add_disturbance(cx, cx, 0.3, 0.5),
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

fn bench_flip_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("flip_step");
    for &n in &[25, 100, 225] {
        let spacing = 1.0 / (n as f64).sqrt();
        let particles: Vec<FluidParticle> = (0..n)
            .map(|i| {
                let x = 0.3 + (i % (n as f64).sqrt() as usize) as f64 * spacing;
                let y = 0.3 + (i / (n as f64).sqrt() as usize) as f64 * spacing;
                FluidParticle::new_2d(x, y, 0.001)
            })
            .collect();
        let mut solver = FlipSolver::new(16, 16, 0.1, 0.95).unwrap();
        group.bench_function(format!("{n}_particles"), |b| {
            b.iter_batched(
                || particles.clone(),
                |mut p| {
                    solver
                        .step(&mut p, DVec3::new(0.0, -9.81, 0.0), 0.01)
                        .unwrap()
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sph_kernels,
    bench_sph_density,
    bench_sph_step,
    bench_sph_pressure_force,
    bench_sph_viscosity_force,
    bench_solver_step,
    bench_solver_step_surface_tension,
    bench_pcisph_step,
    bench_grid_diffuse,
    bench_grid_max_speed,
    bench_grid_step,
    bench_grid_step_maccormack,
    bench_grid_step_periodic,
    bench_shallow_step,
    bench_shallow_volume,
    bench_shallow_disturbance,
    bench_create_particle_block,
    bench_coupling_sph_bodies,
    bench_particle_level_set,
    bench_drag_from_particles,
    bench_flip_step,
);
criterion_main!(benches);
