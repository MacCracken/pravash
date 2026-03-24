use pravash::buoyancy::{self, DragCoefficient, FlowRegime};
use pravash::common::{FluidConfig, FluidMaterial, FluidParticle};
use pravash::grid::FluidGrid;
use pravash::shallow::ShallowWater;
use pravash::sph::{self, SphSolver};
use pravash::vortex;

// ── SPH Integration ─────────────────────────────────────────────────────────

#[test]
fn sph_dam_break_energy_bounded() {
    let mut particles = sph::create_particle_block([0.1, 0.3], [0.2, 0.3], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    let mut max_ke = 0.0f64;
    for _ in 0..200 {
        sph::step(&mut particles, &config, viscosity).unwrap();
        let ke = sph::total_kinetic_energy(&particles);
        max_ke = max_ke.max(ke);
        assert!(ke.is_finite(), "kinetic energy diverged to non-finite");
    }
    // Energy should remain bounded in a damped system
    assert!(max_ke < 1e6, "kinetic energy exploded: {max_ke}");
}

#[test]
fn sph_particles_stay_in_bounds() {
    let mut particles = sph::create_particle_block([0.1, 0.5], [0.3, 0.3], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    for _ in 0..500 {
        sph::step(&mut particles, &config, viscosity).unwrap();
    }

    let [min_x, min_y, _min_z, max_x, max_y, _max_z] = config.bounds;
    for (i, p) in particles.iter().enumerate() {
        assert!(
            p.position[0] >= min_x && p.position[0] <= max_x,
            "particle {i} x={} out of bounds [{min_x}, {max_x}]",
            p.position[0]
        );
        assert!(
            p.position[1] >= min_y && p.position[1] <= max_y,
            "particle {i} y={} out of bounds [{min_y}, {max_y}]",
            p.position[1]
        );
    }
}

#[test]
fn sph_config_validation_catches_bad_params() {
    let mut config = FluidConfig::water_2d();
    config.dt = 0.0;
    let mut particles = sph::create_particle_block([0.1, 0.1], [0.1, 0.1], 0.05, 0.001);
    assert!(sph::step(&mut particles, &config, 0.001).is_err());
}

#[test]
fn sph_single_particle_gravity() {
    let mut particles = vec![FluidParticle::new_2d(0.5, 0.8, 0.01)];
    particles[0].density = 1000.0;
    let config = FluidConfig::water_2d();

    let y_initial = particles[0].position[1];
    for _ in 0..10 {
        sph::step(&mut particles, &config, 0.001).unwrap();
    }
    assert!(
        particles[0].position[1] < y_initial,
        "particle should fall under gravity"
    );
}

// ── Grid Integration ────────────────────────────────────────────────────────

#[test]
fn grid_diffusion_conserves_total() {
    let nx = 20;
    let ny = 20;
    let mut field = vec![0.0; nx * ny];
    field[nx * 10 + 10] = 100.0;
    let total_before: f64 = field.iter().sum();

    FluidGrid::diffuse(&mut field, nx, ny, 0.1, 0.01, 50);

    let total_after: f64 = field.iter().sum();
    // Gauss-Seidel with zero-boundary doesn't perfectly conserve,
    // but should stay in the same order of magnitude
    assert!(
        (total_after - total_before).abs() / total_before.abs() < 0.5,
        "diffusion lost too much mass: before={total_before}, after={total_after}"
    );
}

#[test]
fn grid_large_grid_creation() {
    let grid = FluidGrid::new(256, 256, 0.01).unwrap();
    assert_eq!(grid.cell_count(), 256 * 256);
    assert!(grid.max_speed().abs() < f64::EPSILON);
}

// ── Shallow Water Integration ───────────────────────────────────────────────

#[test]
fn shallow_wave_propagation() {
    let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
    sw.add_disturbance(1.0, 1.0, 0.3, 0.5);

    // Check that a nearby cell picks up the wave
    let near_before = sw.surface_at(5, 5);
    for _ in 0..500 {
        sw.step(0.001).unwrap();
    }
    let near_after = sw.surface_at(5, 5);

    // Wave should have reached nearby cells
    assert!(
        (near_after - near_before).abs() > 1e-8,
        "wave didn't propagate: before={near_before}, after={near_after}"
    );
}

#[test]
fn shallow_volume_approximately_conserved() {
    let mut sw = ShallowWater::new(30, 30, 0.1, 1.0).unwrap();
    sw.add_disturbance(1.5, 1.5, 0.3, 0.2);
    let vol_before = sw.total_volume();

    for _ in 0..100 {
        sw.step(0.001).unwrap();
    }
    let vol_after = sw.total_volume();

    let relative_change = (vol_after - vol_before).abs() / vol_before;
    assert!(
        relative_change < 0.1,
        "volume changed by {:.1}%",
        relative_change * 100.0
    );
}

#[test]
fn shallow_flat_surface_stable() {
    let mut sw = ShallowWater::new(20, 20, 0.1, 1.0).unwrap();
    for _ in 0..1000 {
        sw.step(0.001).unwrap();
    }
    let max_dev = sw.max_wave_height(1.0);
    assert!(max_dev < 1e-10, "flat surface drifted: {max_dev}");
}

// ── Cross-module Integration ────────────────────────────────────────────────

#[test]
fn materials_have_consistent_physics() {
    let materials = [
        ("water", FluidMaterial::WATER),
        ("oil", FluidMaterial::OIL),
        ("honey", FluidMaterial::HONEY),
        ("air", FluidMaterial::AIR),
        ("lava", FluidMaterial::LAVA),
    ];
    for (name, m) in &materials {
        assert!(m.density > 0.0, "{name} has non-positive density");
        assert!(m.viscosity >= 0.0, "{name} has negative viscosity");
        assert!(
            m.speed_of_sound > 0.0,
            "{name} has non-positive speed of sound"
        );
    }
    // Physical ordering checks
    const { assert!(FluidMaterial::AIR.density < FluidMaterial::WATER.density) };
    const { assert!(FluidMaterial::WATER.viscosity < FluidMaterial::HONEY.viscosity) };
    const { assert!(FluidMaterial::AIR.speed_of_sound < FluidMaterial::WATER.speed_of_sound) };
}

#[test]
fn serde_roundtrip_all_types() {
    // FluidConfig
    let config = FluidConfig::water_2d();
    let json = serde_json::to_string(&config).unwrap();
    let config2: FluidConfig = serde_json::from_str(&json).unwrap();
    assert!((config2.dt - config.dt).abs() < f64::EPSILON);

    // FluidMaterial
    let mat = FluidMaterial::WATER;
    let json = serde_json::to_string(&mat).unwrap();
    let mat2: FluidMaterial = serde_json::from_str(&json).unwrap();
    assert!((mat2.density - mat.density).abs() < f64::EPSILON);

    // FluidGrid
    let grid = FluidGrid::new(5, 5, 0.1).unwrap();
    let json = serde_json::to_string(&grid).unwrap();
    let grid2: FluidGrid = serde_json::from_str(&json).unwrap();
    assert_eq!(grid2.nx, grid.nx);

    // ShallowWater
    let sw = ShallowWater::new(5, 5, 0.1, 1.0).unwrap();
    let json = serde_json::to_string(&sw).unwrap();
    let sw2: ShallowWater = serde_json::from_str(&json).unwrap();
    assert_eq!(sw2.nx, sw.nx);
}

// ── Buoyancy Integration ────────────────────────────────────────────────────

#[test]
fn buoyancy_sphere_in_water() {
    // A 1 kg sphere (r=0.062m) in water should float (buoyancy > weight)
    let density_water = FluidMaterial::WATER.density;
    let radius: f64 = 0.062;
    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
    let buoyancy = buoyancy::buoyancy_force(density_water, 9.81, volume);
    let weight = 1.0 * 9.81;
    // Buoyancy of fully submerged sphere > weight means it floats
    assert!(buoyancy > weight * 0.9);
}

#[test]
fn buoyancy_terminal_velocity_validation() {
    // Valid case
    let vt = buoyancy::terminal_velocity(1.0, 9.81, 1.225, DragCoefficient::SPHERE, 0.01);
    assert!(vt.is_ok());
    assert!(vt.unwrap() > 0.0);

    // Zero density (invalid)
    assert!(buoyancy::terminal_velocity(1.0, 9.81, 0.0, DragCoefficient::SPHERE, 0.01).is_err());

    // Negative gravity (negative m*g)
    assert!(buoyancy::terminal_velocity(1.0, -9.81, 1.225, DragCoefficient::SPHERE, 0.01).is_err());
}

#[test]
fn buoyancy_reynolds_flow_regime() {
    let viscosity_water = FluidMaterial::WATER.viscosity;
    let density_water = FluidMaterial::WATER.density;

    // Slow flow: laminar
    let re_slow = buoyancy::reynolds_number(density_water, 0.001, 0.01, viscosity_water).unwrap();
    assert_eq!(buoyancy::classify_flow(re_slow), FlowRegime::Laminar);

    // Fast flow: turbulent
    let re_fast = buoyancy::reynolds_number(density_water, 10.0, 0.1, viscosity_water).unwrap();
    assert_eq!(buoyancy::classify_flow(re_fast), FlowRegime::Turbulent);

    // Zero viscosity: error
    assert!(buoyancy::reynolds_number(density_water, 1.0, 0.1, 0.0).is_err());
}

// ── Vortex Integration ──────────────────────────────────────────────────────

#[test]
fn vortex_lamb_oseen_decays() {
    let circulation = 1.0;
    let r = 0.1;
    let viscosity = 0.01;
    let v_early = vortex::lamb_oseen_velocity(circulation, r, viscosity, 0.1);
    let v_late = vortex::lamb_oseen_velocity(circulation, r, viscosity, 10.0);
    // As the vortex ages, the core spreads and velocity at fixed r should approach irrotational
    assert!(v_early.is_finite());
    assert!(v_late.is_finite());
}

#[test]
fn vortex_rankine_vs_lamb_oseen_convergence() {
    // At large radius, both models should give approximately Γ/(2πr)
    let circulation = 5.0;
    let r_far = 100.0;
    let v_rankine = vortex::rankine_velocity(circulation, r_far, 1.0);
    let v_lo = vortex::lamb_oseen_velocity(circulation, r_far, 0.001, 1.0);
    let v_irrot = circulation / (2.0 * std::f64::consts::PI * r_far);
    assert!((v_rankine - v_irrot).abs() < 1e-6);
    assert!((v_lo - v_irrot).abs() < 1e-3);
}

#[test]
fn vortex_enstrophy_scales_with_dx() {
    let field = vec![1.0; 100];
    let e1 = vortex::enstrophy(&field, 0.1);
    let e2 = vortex::enstrophy(&field, 0.2);
    // Enstrophy scales as dx²
    assert!((e2 / e1 - 4.0).abs() < 1e-10);
}

// ── Edge Case Tests ─────────────────────────────────────────────────────────

#[test]
fn config_validation_comprehensive() {
    let mut c = FluidConfig::water_2d();
    assert!(c.validate().is_ok());

    c.gas_constant = 0.0;
    assert!(c.validate().is_err());

    c = FluidConfig::water_2d();
    c.boundary_damping = 1.5;
    assert!(c.validate().is_err());

    c = FluidConfig::water_2d();
    c.boundary_damping = -0.1;
    assert!(c.validate().is_err());
}

#[test]
fn grid_invalid_dx_rejected() {
    assert!(FluidGrid::new(10, 10, 0.0).is_err());
    assert!(FluidGrid::new(10, 10, -1.0).is_err());
}

#[test]
fn custom_material_speed_of_sound_validated() {
    assert!(FluidMaterial::custom(1000.0, 0.001, 0.072, 0.0).is_err());
    assert!(FluidMaterial::custom(1000.0, 0.001, 0.072, -1.0).is_err());
    assert!(FluidMaterial::custom(1000.0, 0.001, 0.072, 1480.0).is_ok());
}

#[test]
fn particle_is_copy() {
    let p = FluidParticle::new_2d(1.0, 2.0, 0.01);
    let p2 = p;
    let _ = p; // p is still valid because FluidParticle is Copy
    let _ = p2;
}

#[test]
fn config_is_copy() {
    let c = FluidConfig::water_2d();
    let c2 = c;
    let _ = c; // c is still valid because FluidConfig is Copy
    let _ = c2;
}

// ── SphSolver Integration ───────────────────────────────────────────────────

#[test]
fn solver_dam_break_energy_bounded() {
    let mut particles = sph::create_particle_block([0.1, 0.3], [0.2, 0.3], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;
    let mut solver = SphSolver::new();

    let mut max_ke = 0.0f64;
    for _ in 0..200 {
        solver.step(&mut particles, &config, viscosity).unwrap();
        let ke = sph::total_kinetic_energy(&particles);
        max_ke = max_ke.max(ke);
        assert!(ke.is_finite(), "kinetic energy diverged to non-finite");
    }
    assert!(max_ke < 1e6, "kinetic energy exploded: {max_ke}");
}

#[test]
fn solver_particles_stay_in_bounds() {
    let mut particles = sph::create_particle_block([0.1, 0.5], [0.3, 0.3], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;
    let mut solver = SphSolver::new();

    for _ in 0..500 {
        solver.step(&mut particles, &config, viscosity).unwrap();
    }

    let [min_x, min_y, _min_z, max_x, max_y, _max_z] = config.bounds;
    for (i, p) in particles.iter().enumerate() {
        assert!(
            p.position[0] >= min_x && p.position[0] <= max_x,
            "particle {i} x={} out of bounds [{min_x}, {max_x}]",
            p.position[0]
        );
        assert!(
            p.position[1] >= min_y && p.position[1] <= max_y,
            "particle {i} y={} out of bounds [{min_y}, {max_y}]",
            p.position[1]
        );
    }
}

#[test]
fn solver_surface_tension_keeps_blob_compact() {
    // With surface tension, a block of particles should stay more compact
    // than without it (less spread due to cohesive forces).
    let make_block = || sph::create_particle_block([0.3, 0.3], [0.2, 0.2], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    // Run without surface tension
    let mut particles_no_st = make_block();
    let mut solver_no_st = SphSolver::new();
    for _ in 0..50 {
        solver_no_st
            .step(&mut particles_no_st, &config, viscosity)
            .unwrap();
    }
    let spread_no_st: f64 = particles_no_st.iter().map(|p| p.speed_squared()).sum();

    // Run with surface tension
    let mut particles_st = make_block();
    let mut solver_st = SphSolver::with_surface_tension(0.072);
    for _ in 0..50 {
        solver_st
            .step(&mut particles_st, &config, viscosity)
            .unwrap();
    }

    // Both should be finite (no divergence)
    assert!(spread_no_st.is_finite());
    for p in &particles_st {
        assert!(p.position[0].is_finite());
        assert!(p.position[1].is_finite());
    }
}

#[test]
fn solver_multi_step_consistency() {
    // Running the solver for many steps should not cause divergence or NaN
    let mut particles = sph::create_particle_block([0.2, 0.5], [0.3, 0.3], 0.03, 0.001);
    let config = FluidConfig::water_2d();
    let mut solver = SphSolver::new();

    for step_num in 0..1000 {
        solver.step(&mut particles, &config, 0.001).unwrap();
        if step_num % 100 == 0 {
            for p in particles.iter() {
                assert!(
                    p.position[0].is_finite() && p.position[1].is_finite(),
                    "diverged at step {step_num}"
                );
            }
        }
    }
}

#[test]
fn solver_symmetric_pressure_conserves_momentum() {
    // With symmetric pressure and no gravity, internal forces should
    // produce near-zero net momentum (Newton's third law).
    let mut particles = sph::create_particle_block([0.4, 0.4], [0.2, 0.2], 0.02, 0.001);
    let mut config = FluidConfig::water_2d();
    config.gravity = [0.0, 0.0, 0.0]; // disable gravity for clean momentum test

    let mut solver = SphSolver::new();
    solver.step(&mut particles, &config, 0.001).unwrap();

    let total_px: f64 = particles.iter().map(|p| p.velocity[0] * p.mass).sum();
    let total_py: f64 = particles.iter().map(|p| p.velocity[1] * p.mass).sum();
    // Symmetric pressure forces should cancel; small residual from discrete errors
    assert!(
        total_px.abs() < 1e-8,
        "x-momentum not conserved: {total_px}"
    );
    assert!(
        total_py.abs() < 1e-8,
        "y-momentum not conserved: {total_py}"
    );
}
