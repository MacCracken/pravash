//! `bridge.tsv` — `rust-old/src/bridge.rs` (M1 part 4): the ten pavan / ushma / goonj conversions over their physical
//! ranges, edges and the specials. `temperature_to_water_viscosity` (powf) and `tke_to_broadband_noise_db` (log10)
//! are class C, gated at 0 ulp against the corrected oracle (ADR 0004); `wind_to_freestream` (a clamp) is class A;
//! the rest is class B. `aero_coefficients_to_forces` gives (lift, drag).

use pravash::bridge::{
    aero_coefficients_to_forces, cp_to_surface_pressure, freestream_dynamic_pressure,
    heat_to_buoyancy_forcing, temperature_to_water_viscosity, tke_to_broadband_noise_db,
    tke_to_eddy_diffusivity, velocity_to_acoustic_source_strength,
    wave_height_to_acoustic_pressure, wind_to_freestream,
};

use crate::grids::{Rng, specials};
use crate::tsv::{Class, Module, f};

fn grid(
    base: &[f64],
    rng: &mut Rng,
    k: usize,
    draw: &dyn Fn(&mut Rng, usize) -> f64,
) -> Vec<Vec<f64>> {
    let mut out = vec![base.to_vec()];
    for a in 0..base.len() {
        for &x in &specials() {
            let mut c = base.to_vec();
            c[a] = x;
            out.push(c);
        }
        for &x in &[-1.0, 0.5, 1e-300, 1e300] {
            let mut c = base.to_vec();
            c[a] = x;
            out.push(c);
        }
    }
    for _ in 0..k {
        out.push((0..base.len()).map(|a| draw(rng, a)).collect());
    }
    out
}

#[allow(clippy::too_many_lines)] // one block per conversion
pub fn emit() -> Module {
    let mut m = Module::new("bridge");
    let mut rng = Rng::new(0xb41d6e);
    let wide = |r: &mut Rng, _: usize| {
        if r.unit() < 0.7 {
            r.uniform(-500.0, 500.0)
        } else {
            r.signed_log_uniform(-100.0, 100.0)
        }
    };
    for c in grid(&[-0.5, 600.0, 101_325.0], &mut rng, 200, &wide) {
        m.row(
            "cp_to_surface_pressure",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2])],
            &[f(cp_to_surface_pressure(c[0], c[1], c[2]))],
        );
    }
    for c in grid(&[30.0, 1.225], &mut rng, 200, &wide) {
        m.row(
            "freestream_dynamic_pressure",
            Class::B,
            &[f(c[0]), f(c[1])],
            &[f(freestream_dynamic_pressure(c[0], c[1]))],
        );
    }
    for c in grid(&[1.2, 0.05, 551.25, 16.0], &mut rng, 200, &wide) {
        let (l, d) = aero_coefficients_to_forces(c[0], c[1], c[2], c[3]);
        m.row(
            "aero_coefficients_to_forces",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3])],
            &[f(l), f(d)],
        );
    }
    let mut winds = specials();
    for x in [340.0f64, -340.0] {
        winds.extend([
            x,
            f64::from_bits(x.to_bits() - 1),
            f64::from_bits(x.to_bits() + 1),
        ]);
    }
    for _ in 0..200 {
        winds.push(rng.uniform(-500.0, 500.0));
    }
    for &x in &winds {
        m.row(
            "wind_to_freestream",
            Class::A,
            &[f(x)],
            &[f(wind_to_freestream(x))],
        );
    }
    // The clamp's ends, around them, every integer kelvin of the valid range, and a sweep.
    let mut temps = specials();
    temps.extend([
        273.0,
        373.0,
        272.999_999_999_999_94,
        373.000_000_000_000_06,
        200.0,
        500.0,
        293.15,
    ]);
    temps.extend((273..=373).map(f64::from));
    for _ in 0..700 {
        temps.push(rng.uniform(260.0, 390.0));
    }
    for &x in &temps {
        m.row(
            "temperature_to_water_viscosity",
            Class::C,
            &[f(x)],
            &[f(temperature_to_water_viscosity(x))],
        );
    }
    let heat = |r: &mut Rng, a: usize| match a {
        0 => r.uniform(-1000.0, 10_000.0),
        1 => r.log_uniform(-6.0, 0.0),
        2 => r.uniform(-10.0, 2000.0),
        3 => r.uniform(-100.0, 5000.0),
        4 => r.log_uniform(-5.0, -3.0),
        _ => r.log_uniform(-4.0, -1.0),
    };
    for c in grid(
        &[1000.0, 0.001, 1000.0, 4186.0, 0.00021, 0.01],
        &mut rng,
        300,
        &heat,
    ) {
        m.row(
            "heat_to_buoyancy_forcing",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2]), f(c[3]), f(c[4]), f(c[5])],
            &[f(heat_to_buoyancy_forcing(
                c[0], c[1], c[2], c[3], c[4], c[5],
            ))],
        );
    }
    let tke = |r: &mut Rng, a: usize| match a {
        0 => r.uniform(-1.0, 10.0),
        _ => r.log_uniform(-4.0, 1.0),
    };
    for c in grid(&[0.5, 0.1], &mut rng, 300, &tke) {
        m.row(
            "tke_to_eddy_diffusivity",
            Class::B,
            &[f(c[0]), f(c[1])],
            &[f(tke_to_eddy_diffusivity(c[0], c[1]))],
        );
    }
    let mach = |r: &mut Rng, a: usize| match a {
        0 => r.uniform(-400.0, 400.0),
        _ => r.uniform(-10.0, 1500.0),
    };
    for c in grid(&[30.0, 343.0], &mut rng, 300, &mach) {
        m.row(
            "velocity_to_acoustic_source_strength",
            Class::B,
            &[f(c[0]), f(c[1])],
            &[f(velocity_to_acoustic_source_strength(c[0], c[1]))],
        );
    }
    // Broadband noise: k around the 0 dB edge (u_rms = 5e-8 at k = 3.75e-15), up to violent turbulence.
    let mut ks = specials();
    ks.extend([
        3.75e-15,
        3.75e-15 * (1.0 + f64::EPSILON),
        3.75e-15 * (1.0 - f64::EPSILON),
        1e-300,
        1e300,
    ]);
    for _ in 0..800 {
        ks.push(rng.log_uniform(-16.0, 4.0));
    }
    for &x in &ks {
        m.row(
            "tke_to_broadband_noise_db",
            Class::C,
            &[f(x)],
            &[f(tke_to_broadband_noise_db(x))],
        );
    }
    let wave = |r: &mut Rng, a: usize| match a {
        0 => r.uniform(-5.0, 5.0),
        1 => r.uniform(-1.0, 20_000.0),
        _ => r.uniform(-10.0, 1500.0),
    };
    for c in grid(&[0.5, 1.0, 343.0], &mut rng, 300, &wave) {
        m.row(
            "wave_height_to_acoustic_pressure",
            Class::B,
            &[f(c[0]), f(c[1]), f(c[2])],
            &[f(wave_height_to_acoustic_pressure(c[0], c[1], c[2]))],
        );
    }
    m
}
