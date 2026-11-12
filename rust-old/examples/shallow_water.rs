use pravash::shallow::ShallowWater;

fn main() {
    let mut sw = ShallowWater::new(40, 40, 0.1, 1.0).unwrap();
    sw.manning_n.fill(0.025); // earth channel friction

    // Drop a splash in the center
    sw.add_disturbance(2.0, 2.0, 0.3, 0.5);

    println!("Pravash — Shallow water simulation");
    println!("Grid: {}x{}, dx={}", sw.nx, sw.ny, sw.dx);
    println!();

    for frame in 0..200 {
        sw.step(0.001).unwrap();

        if frame % 20 == 0 {
            let vol = sw.total_volume();
            let max_h = sw.max_wave_height(1.0);
            println!("Frame {frame:>4}  volume: {vol:>8.4}  max_wave: {max_h:>8.6}");
        }
    }

    println!("\nDone.");
}
