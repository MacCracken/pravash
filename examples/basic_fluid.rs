use pravash::common::{FluidConfig, FluidMaterial};
use pravash::sph::{create_particle_block, max_speed, step, total_kinetic_energy};

fn main() {
    let mut particles = create_particle_block([0.2, 0.5], [0.3, 0.3], 0.02, 0.001);
    let config = FluidConfig::water_2d();
    let viscosity = FluidMaterial::WATER.viscosity;

    println!("Pravash — SPH fluid simulation");
    println!("Particles: {}", particles.len());
    println!("Timestep:  {} s", config.dt);
    println!();

    for frame in 0..100 {
        step(&mut particles, &config, viscosity).unwrap();

        if frame % 10 == 0 {
            let ke = total_kinetic_energy(&particles);
            let ms = max_speed(&particles);
            println!("Frame {frame:>4}  KE: {ke:>10.4}  max_speed: {ms:>8.4}",);
        }
    }

    println!();
    println!("Done.");
}
