use pravash::common::FluidConfig;
use pravash::sph::{
    MultiPhaseConfig, PhaseProperties, SphSolver, create_particle_block, max_speed,
};

fn main() {
    let config = FluidConfig::water_2d();
    let phase_config = MultiPhaseConfig::two_phase(
        PhaseProperties {
            rest_density: 1000.0,
            gas_constant: 2000.0,
            viscosity: 0.001,
        },
        PhaseProperties {
            rest_density: 800.0,
            gas_constant: 1500.0,
            viscosity: 0.03,
        },
        0.05, // interface tension
    );

    // Water block (phase 0, bottom)
    let mut particles = create_particle_block([0.2, 0.1], [0.3, 0.2], 0.02, 0.001);
    // Oil block (phase 1, top)
    let mut oil = create_particle_block([0.2, 0.3], [0.3, 0.2], 0.02, 0.0008);
    for p in oil.iter_mut() {
        p.phase = 1;
    }
    particles.extend(oil);

    let mut solver = SphSolver::new();

    println!("Pravash — Multi-phase SPH (water + oil)");
    println!("Particles: {} (water + oil)", particles.len());
    println!();

    for frame in 0..100 {
        solver
            .step_multiphase(&mut particles, &config, &phase_config)
            .unwrap();

        if frame % 10 == 0 {
            let ms = max_speed(&particles);
            let water_count = particles.iter().filter(|p| p.phase == 0).count();
            let oil_count = particles.iter().filter(|p| p.phase == 1).count();
            println!(
                "Frame {frame:>4}  water: {water_count}  oil: {oil_count}  max_speed: {ms:>8.4}"
            );
        }
    }

    println!("\nDone.");
}
