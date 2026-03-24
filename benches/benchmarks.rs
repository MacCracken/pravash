use criterion::{Criterion, black_box, criterion_group, criterion_main};

use pravash::common::{FluidConfig, FluidMaterial};
use pravash::grid::FluidGrid;
use pravash::shallow::ShallowWater;
use pravash::sph;

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
                |mut f| FluidGrid::diffuse(&mut f, n, n, 0.1, 0.01, 20),
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

criterion_group!(
    benches,
    bench_sph_kernels,
    bench_sph_density,
    bench_sph_step,
    bench_sph_pressure_force,
    bench_grid_diffuse,
    bench_grid_max_speed,
    bench_shallow_step,
    bench_shallow_volume,
    bench_create_particle_block,
);
criterion_main!(benches);
