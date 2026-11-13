//! `json.tsv` — the M1 types' serde_json documents (port plan D4, ADR 0002 #6 and #10): `FluidMaterial`,
//! `FluidParticle`, `FluidConfig`, `ParticleSoa` (common.rs), `FlowRegime` and `DragCoefficient` (buoyancy.rs).
//!
//! Each row's input is one document (`s:`); its output is what serde_json 1.0.154 (with `float_roundtrip`) reads
//! from it: the value's tokens (m_common.rs's layouts; a `ParticleSoa` is 18 `v:<n>` columns, the phase as `i:`; a
//! `FlowRegime` is `e:FlowRegime.<Variant>`; the unit struct `DragCoefficient` has no tokens), or `E:Deserialize`
//! when serde_json refuses it. The documents are serde_json's own text for a sweep of values (NaN and ±∞ are written
//! `null`, which it then refuses), and hand-written variants: the array form, unknown, missing and wrong-typed
//! fields, vector lengths, the phase's u8 range, number edges, enum forms.
//!
//! Two kinds of document are emitted under their own names, which the harness leaves pending: those bayan refuses
//! and serde_json reads (an integer above i64::MAX into an f64: `<T>::from_json_bayan_s3`; a duplicated unknown
//! field: `<T>::from_json_bayan_s4`, impetus port plan §8; the integer `-0`, which bayan reads as +0 and serde_json as
//! −0.0: `<T>::from_json_bayan_neg_zero`). The text the port writes is not compared (D4): the harness round-trips its
//! own encoding instead.

use hisab::DVec3;
use pravash::buoyancy::{DragCoefficient, FlowRegime};
use pravash::{FluidConfig, FluidMaterial, FluidParticle, ParticleSoa};
use serde::de::DeserializeOwned;

use crate::grids::{Rng, specials};
use crate::m_common::{config, material, particle};
use crate::tsv::{Class, Module, Tok, e, f, i, s};

pub fn soa_toks(soa: &ParticleSoa) -> Vec<Tok> {
    let mut out = Vec::new();
    let cols: [&Vec<f64>; 17] = [
        &soa.pos_x,
        &soa.pos_y,
        &soa.pos_z,
        &soa.vel_x,
        &soa.vel_y,
        &soa.vel_z,
        &soa.accel_x,
        &soa.accel_y,
        &soa.accel_z,
        &soa.density,
        &soa.pressure,
        &soa.mass,
        &soa.conf_xx,
        &soa.conf_xy,
        &soa.conf_yy,
        &soa.temperature,
        &soa.fuel,
    ];
    for (k, col) in cols.iter().enumerate() {
        if k == 12 {
            out.push(Tok::V(soa.phase.len()));
            out.extend(soa.phase.iter().map(|&p| i(p)));
        }
        out.push(Tok::V(col.len()));
        out.extend(col.iter().map(|&x| f(x)));
    }
    out
}

/// One decode row: `doc` read as T by serde_json, its tokens or `E:Deserialize`.
fn row<T: DeserializeOwned>(m: &mut Module, name: &str, doc: &str, toks: &dyn Fn(&T) -> Vec<Tok>) {
    let out = match serde_json::from_str::<T>(doc) {
        Ok(v) => toks(&v),
        Err(_) => vec![Tok::Err("Deserialize".to_string())],
    };
    m.row(name, Class::A, &[s(doc)], &out);
}

fn material_rows(m: &mut Module, rng: &mut Rng) {
    let name = "FluidMaterial::from_json";
    let tk = |v: &FluidMaterial| material(v);
    let mut vals = vec![
        FluidMaterial::WATER,
        FluidMaterial::OIL,
        FluidMaterial::HONEY,
        FluidMaterial::AIR,
        FluidMaterial::LAVA,
    ];
    for &x in &specials() {
        let mut v = FluidMaterial::WATER;
        v.viscosity = x;
        vals.push(v);
    }
    for _ in 0..60 {
        let mut v = FluidMaterial::WATER;
        v.density = rng.signed_log_uniform(-300.0, 300.0);
        v.viscosity = rng.uniform(0.0, 1.0);
        v.surface_tension = rng.log_uniform(-20.0, 0.0);
        v.speed_of_sound = rng.uniform(100.0, 3000.0);
        vals.push(v);
    }
    for v in &vals {
        row(m, name, &serde_json::to_string(v).unwrap_or_default(), &tk);
    }
    let w = r#""density":1000.0,"viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0"#;
    for doc in [
        format!("{{{w}}}"),
        format!("{{{w},\"colour\":\"blue\"}}"),
        format!(" {{ {w} }} \n"),
        "[1000.0,0.001,0.072,1480.0]".to_string(),
        "[1000,0,0,1480]".to_string(),
        "[1000.0,0.001,0.072]".to_string(),
        "[1000.0,0.001,0.072,1480.0,5.0]".to_string(),
        r#"{"density":1000.0,"viscosity":0.001,"surface_tension":0.072}"#.to_string(),
        r#"{"density":"1000","viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0}"#.to_string(),
        r#"{"density":null,"viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0}"#.to_string(),
        r#"{"density":true,"viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0}"#.to_string(),
        r#"{"density":-0.0,"viscosity":0,"surface_tension":0.0,"speed_of_sound":1e3}"#.to_string(),
        r#"{"density":1e400,"viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0}"#.to_string(),
        r#"{"density":1e-400,"viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0}"#.to_string(),
        r#"{"density":2.2250738585072011e-308,"viscosity":4.9e-324,"surface_tension":1.7976931348623157e308,"speed_of_sound":0.1E+2}"#.to_string(),
        r#"{"density":1000.0,"density":999.0,"viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0}"#.to_string(),
        format!("{{{w}}} x"),
        format!("{{{w}"),
        "{}".to_string(),
        "[]".to_string(),
        "null".to_string(),
        "1000".to_string(),
        String::new(),
    ] {
        row(m, name, &doc, &tk);
    }
    // bayan's stricter policies (S3, S4): pending in the harness.
    row(
        m,
        "FluidMaterial::from_json_bayan_s3",
        r#"{"density":100000000000000000000,"viscosity":0.001,"surface_tension":0.072,"speed_of_sound":1480.0}"#,
        &tk,
    );
    row(
        m,
        "FluidMaterial::from_json_bayan_s4",
        &format!("{{{w},\"x\":1,\"x\":2}}"),
        &tk,
    );
    // bayan reads the integer -0 as 0, losing the sign serde_json keeps (-0.0): pending too.
    row(
        m,
        "FluidMaterial::from_json_bayan_neg_zero",
        r#"{"density":1000.0,"viscosity":-0,"surface_tension":0.072,"speed_of_sound":1480.0}"#,
        &tk,
    );
}

fn particle_rows(m: &mut Module, rng: &mut Rng) {
    let name = "FluidParticle::from_json";
    let tk = |v: &FluidParticle| particle(v);
    let sp = specials();
    let mut vals = vec![FluidParticle::new_2d(0.0, 0.0, 0.001)];
    for n in 0..80 {
        let mut p = FluidParticle::new_2d(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.log_uniform(-6.0, 0.0),
        );
        p.velocity = DVec3::new(
            rng.uniform(-10.0, 10.0),
            rng.uniform(-10.0, 10.0),
            rng.signed_log_uniform(-30.0, 30.0),
        );
        p.acceleration = DVec3::new(rng.uniform(-100.0, 100.0), -9.81, 0.0);
        p.density = rng.uniform(900.0, 1100.0);
        p.pressure = rng.uniform(-1000.0, 1000.0);
        p.phase = u8::try_from(n * 37 % 256).unwrap_or(0);
        p.conformation = [
            rng.uniform(0.5, 1.5),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.5, 1.5),
        ];
        p.temperature = rng.uniform(250.0, 400.0);
        p.fuel = rng.unit();
        if n < sp.len() {
            p.pressure = sp[n];
        }
        vals.push(p);
    }
    for v in &vals {
        row(m, name, &serde_json::to_string(v).unwrap_or_default(), &tk);
    }
    let rest = r#""velocity":[0.5,-0.5,0.0],"acceleration":[0.0,-9.81,0.0],"density":1000.0,"pressure":0.0,"mass":0.001"#;
    let pre = format!("\"position\":[1.0,2.0,0.0],{rest}");
    let post = r#""conformation":[1.0,0.0,1.0],"temperature":293.15,"fuel":0.0"#;
    for (ph, extra) in [
        ("0", ""),
        ("255", ""),
        ("256", ""),
        ("-1", ""),
        ("1.0", ""),
        ("3", ",\"spin\":[1,2]"),
    ] {
        row(
            m,
            name,
            &format!("{{{pre},\"phase\":{ph},{post}{extra}}}"),
            &tk,
        );
    }
    for doc in [
        r#"[[1.0,2.0,0.0],[0.5,-0.5,0.0],[0.0,-9.81,0.0],1000.0,0.0,0.001,7,[1.0,0.0,1.0],293.15,0.0]"#.to_string(),
        r#"[[1.0,2.0,0.0],[0.5,-0.5,0.0],[0.0,-9.81,0.0],1000.0,0.0,0.001,7,[1.0,0.0,1.0],293.15]"#.to_string(),
        format!("{{{pre},\"phase\":0,\"conformation\":[1.0,0.0],\"temperature\":293.15,\"fuel\":0.0}}"),
        format!("{{{pre},\"phase\":0,\"conformation\":[1.0,0.0,1.0,0.0],\"temperature\":293.15,\"fuel\":0.0}}"),
        format!("{{\"position\":[1.0,2.0],{rest},\"phase\":0,{post}}}"),
        format!("{{\"position\":{{\"x\":1.0,\"y\":2.0,\"z\":0.0}},{rest},\"phase\":0,{post}}}"),
        format!("{{{pre},{post}}}"),
        format!("{{{pre},\"phase\":0,{post},\"phase\":1}}"),
    ] {
        row(m, name, &doc, &tk);
    }
}

fn config_rows(m: &mut Module, rng: &mut Rng) {
    let name = "FluidConfig::from_json";
    let tk = |v: &FluidConfig| config(v);
    let mut vals = vec![FluidConfig::water_2d()];
    for &x in &specials() {
        let mut c = FluidConfig::water_2d();
        c.gravity.z = x;
        vals.push(c);
    }
    for _ in 0..60 {
        let mut c = FluidConfig::water_2d();
        c.dt = rng.log_uniform(-6.0, -1.0);
        c.gravity = DVec3::new(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-20.0, 0.0),
            rng.uniform(-1.0, 1.0),
        );
        c.smoothing_radius = rng.log_uniform(-3.0, 0.0);
        c.gas_constant = rng.uniform(1.0, 5000.0);
        c.rest_density = rng.uniform(1.0, 3000.0);
        c.bounds_max = DVec3::new(
            rng.uniform(0.5, 10.0),
            rng.uniform(0.5, 10.0),
            rng.uniform(0.0, 10.0),
        );
        c.boundary_damping = rng.unit();
        vals.push(c);
    }
    for v in &vals {
        row(m, name, &serde_json::to_string(v).unwrap_or_default(), &tk);
    }
    for doc in [
        "[0.001,[0,-9.81,0],0.05,2000,1000,[0,0,0],[1,1,0],0.5]",
        "[0.001,[0,-9.81,0],0.05,2000,1000,[0,0,0],[1,1,0]]",
        "[0.001,[0,-9.81],0.05,2000,1000,[0,0,0],[1,1,0],0.5]",
        r#"{"dt":0.001,"gravity":[0,-9.81,0],"smoothing_radius":0.05,"gas_constant":2000,"rest_density":1000,"bounds_min":[0,0,0],"bounds_max":[1,1,0],"boundary_damping":0.5,"solver":"pcisph"}"#,
        r#"{"dt":0.001,"gravity":[0,-9.81,0],"smoothing_radius":0.05,"gas_constant":2000,"rest_density":1000,"bounds_min":[0,0,0],"bounds_max":[1,1,0]}"#,
        r#"{"dt":0.001,"gravity":[0,"-9.81",0],"smoothing_radius":0.05,"gas_constant":2000,"rest_density":1000,"bounds_min":[0,0,0],"bounds_max":[1,1,0],"boundary_damping":0.5}"#,
    ] {
        row(m, name, doc, &tk);
    }
}

fn soa_rows(m: &mut Module, rng: &mut Rng) {
    let name = "ParticleSoa::from_json";
    let tk = |v: &ParticleSoa| soa_toks(v);
    for n in 0..6 {
        let ps: Vec<FluidParticle> = (0..n)
            .map(|k| {
                let mut p =
                    FluidParticle::new_2d(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.001);
                p.phase = u8::try_from(k).unwrap_or(0);
                p.velocity.x = rng.signed_log_uniform(-10.0, 10.0);
                p
            })
            .collect();
        let mut soa = ParticleSoa::from_aos(&ps);
        if n == 5 {
            soa.mass.truncate(2);
            soa.fuel.push(0.25);
        }
        row(
            m,
            name,
            &serde_json::to_string(&soa).unwrap_or_default(),
            &tk,
        );
    }
    let empty = serde_json::to_string(&ParticleSoa::new()).unwrap_or_default();
    row(
        m,
        name,
        &empty.replace(r#""phase":[]"#, r#""phase":[1.5]"#),
        &tk,
    );
    row(
        m,
        name,
        &empty.replace(r#""phase":[]"#, r#""phase":[300]"#),
        &tk,
    );
    row(
        m,
        name,
        &empty.replace(r#""fuel":[]"#, r#""fuel":[null]"#),
        &tk,
    );
    row(m, name, &empty.replace(r#","fuel":[]"#, ""), &tk);
    row(
        m,
        name,
        &empty.replace(r#""pos_x":[]"#, r#""pos_x":[1,2.5,-0.0]"#),
        &tk,
    );
}

fn enum_rows(m: &mut Module) {
    let tk = |v: &FlowRegime| vec![e("FlowRegime", v)];
    for v in [
        FlowRegime::Laminar,
        FlowRegime::Transitional,
        FlowRegime::Turbulent,
    ] {
        row(
            m,
            "FlowRegime::from_json",
            &serde_json::to_string(&v).unwrap_or_default(),
            &tk,
        );
    }
    for doc in [
        r#"{"Laminar":null}"#,
        r#"{"Turbulent":null}"#,
        r#"{"Laminar":1}"#,
        r#"{"Laminar":null,"Turbulent":null}"#,
        r#""laminar""#,
        r#""Viscous""#,
        "0",
        "null",
        r#"["Laminar"]"#,
    ] {
        row(m, "FlowRegime::from_json", doc, &tk);
    }
    let tu = |_: &DragCoefficient| vec![];
    let unit = serde_json::to_string(&DragCoefficient).unwrap_or_default();
    for doc in [
        unit.as_str(),
        "[]",
        "{}",
        "0",
        r#""DragCoefficient""#,
        " null ",
    ] {
        row(m, "DragCoefficient::from_json", doc, &tu);
    }
}

pub fn emit() -> Module {
    let mut m = Module::new("json");
    let mut rng = Rng::new(0x15_0a);
    material_rows(&mut m, &mut rng);
    particle_rows(&mut m, &mut rng);
    config_rows(&mut m, &mut rng);
    soa_rows(&mut m, &mut rng);
    enum_rows(&mut m);
    m
}
