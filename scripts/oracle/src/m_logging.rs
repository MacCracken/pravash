//! `logging.tsv` — `rust-old/src/logging.rs` (M7 part b): the `PRAVASH_LOG` filter `init_with_level` builds.
//!
//! `init_with_level(default)` filters with `EnvFilter::try_from_env("PRAVASH_LOG")`, or `EnvFilter::new(default)`
//! when that fails (the variable unset, or a directive that doesn't parse). A row runs the same two constructors on
//! its value (`EnvFilter::builder().parse`, which `try_from_env` calls on the variable's text) and asks the filter, for
//! each of pravash's tracing targets in name order (`pravash::common` … `pravash::vof`), the most verbose level an
//! event of that target is enabled at, outside any span: `i:0` (off) … `i:5` (trace).
//!
//! `init_with_level` takes `s:<PRAVASH_LOG>`, or `n:` for unset, and `s:<default>`, and gives the eight levels (class
//! A). The values exercise EnvFilter's static grammar: level names in any case, the numbers 0–5 (with `+` and leading
//! zeros), targets and their prefixes, the most specific target winning and the last of two equal ones, empty
//! pieces, whitespace (which `Directive::parse` trims for its walk but not for its slices), invalid directives, and
//! the span and field forms, which decide nothing for a field-less callsite outside a span.
//!
//! The trace rows (`trace:<call>`) record what one traced call emits under a subscriber that enables everything, as
//! the port's harness records it from sakshi: `s:>name{k=v …}` when a span is entered (`>name` without fields),
//! `s:<name` when it is exited, and `s:<LEVEL> <target>: <message> k=v …` for an event (class A). Each takes the
//! few numbers its scenario is built from (listed with the scenario), and tests/parity.tcyr builds the same one.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

use hisab::{DVec3, SpatialHash, Vec3};
use pravash::FluidConfig;
use pravash::FluidParticle;
use pravash::common::ParticleArena;
use pravash::coupling::{
    BodyShape, FlipSolver, RigidBody, couple_sph_bodies, drag_from_particles,
    drag_from_particles_indexed, particle_level_set,
};
use pravash::grid::{
    BoundaryCondition, FluidGrid, GridConfig, KEpsilon, MacGrid, apply_ghost_fluid,
};
use pravash::mpm::{ConstitutiveModel, MpmParticle, MpmSolver};
use pravash::phase_field::PhaseField;
use pravash::shallow::{SedimentConfig, ShallowWater, update_sediment};
use pravash::sph::{
    CombustionConfig, HeatConfig, MultiPhaseConfig, PhaseChangeConfig, SphSolver,
    ViscoelasticConfig, apply_contact_angle, apply_delta_sph, apply_implicit_viscosity,
    compute_gradient_corrections, generate_secondary_particles, merge_particles, pressure_force,
    sort_by_zorder, split_particle, update_combustion, update_heat, update_phase_change,
    update_reaction, update_viscoelastic, viscosity_force,
};
use pravash::vof::VofField;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

use crate::m_grid::usz;
use crate::tsv::{Class, Module, Tok, f, i, s};

/// The most verbose level an event of the literal target `$t` is enabled at, under the current default subscriber.
macro_rules! max_level {
    ($t:literal) => {{
        if tracing::event_enabled!(target: $t, Level::TRACE) {
            5
        } else if tracing::event_enabled!(target: $t, Level::DEBUG) {
            4
        } else if tracing::event_enabled!(target: $t, Level::INFO) {
            3
        } else if tracing::event_enabled!(target: $t, Level::WARN) {
            2
        } else if tracing::event_enabled!(target: $t, Level::ERROR) {
            1
        } else {
            0
        }
    }};
}

/// The eight targets' levels under `filter`.
fn levels(filter: EnvFilter) -> Vec<i64> {
    let sub = tracing_subscriber::registry().with(filter);
    tracing::subscriber::with_default(sub, || {
        vec![
            max_level!("pravash::common"),
            max_level!("pravash::coupling"),
            max_level!("pravash::grid"),
            max_level!("pravash::mpm"),
            max_level!("pravash::phase_field"),
            max_level!("pravash::shallow"),
            max_level!("pravash::sph"),
            max_level!("pravash::vof"),
        ]
    })
}

/// `init_with_level`'s filter for the variable `env` (None: unset) and the default.
fn filter(env: Option<&str>, default: &str) -> EnvFilter {
    env.and_then(|v| EnvFilter::builder().parse(v).ok())
        .unwrap_or_else(|| EnvFilter::new(default))
}

const ENVS: &[&str] = &[
    "",
    "info",
    "INFO",
    "Trace",
    "wArN",
    "error",
    "debug",
    "off",
    "0",
    "1",
    "3",
    "5",
    "6",
    "+4",
    "05",
    "-1",
    "pravash=debug",
    "pravash::sph=trace",
    "pravash::sph=trace,pravash=warn",
    "pravash=warn,pravash::sph=trace",
    "pravash::s=debug",
    "pravash::sp=trace,pravash::s=error",
    "pravash::sphx=trace",
    "pravash::sph",
    "pravash",
    "pravash=",
    "pravash::grid=off,info",
    "info,pravash::grid=off",
    "warn=trace",
    "pravash=info,pravash=trace",
    "pravash=trace,pravash=info",
    "trace,pravash::common=error",
    "other=trace",
    "PRAVASH=debug",
    " info",
    "info ",
    " pravash=debug",
    "pravash = debug",
    "pravash=debug ",
    "info,,warn",
    ",",
    " , ",
    "info,bogus!",
    "bogus!",
    "=info",
    "[sph::step]=trace",
    "pravash[sph::step]=trace",
    "pravash[sph::step]",
    "[{n}]=trace",
    "pravash[{n}]=debug",
    "pravash{n=1}=trace",
    "[]",
    "pravash[",
    "pravash[{",
    "pravash[{}]",
    "pravash[x]y",
    "pravash=verbose",
    "pravash=info,other=debug",
    "pravash::grid=debug,pravash::grid=error",
    "debug,pravash::vof=trace,pravash::common=0",
    "pravash::phase_field=trace",
    "pravash::mpm=4,pravash::coupling=2",
    "_x=1,-y=2,:z=3",
];

const DEFAULTS: &[&str] = &[
    "info",
    "trace",
    "warn",
    "garbage!",
    "",
    "pravash::sph=debug",
    "warn,bad!,pravash=error",
    "[sph::step]=trace",
];

/// A subscriber that enables everything and records each span's name and fields, its enters and exits, and each
/// event, in order.
#[derive(Default)]
struct Capture {
    spans: Mutex<Vec<(String, String)>>,
    log: Mutex<Vec<String>>,
}

/// A span's or event's fields as `k=v k=v` (each value's Debug, as tracing's fmt layer writes them), and the message.
#[derive(Default)]
struct Fields {
    message: String,
    rest: String,
}

impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        } else {
            if !self.rest.is_empty() {
                self.rest.push(' ');
            }
            let _ = write!(self.rest, "{}={value:?}", field.name());
        }
    }
}

impl Capture {
    fn span(&self, id: &Id) -> (String, String) {
        let spans = self
            .spans
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let k = usize::try_from(id.into_u64()).unwrap_or(0);
        spans.get(k.wrapping_sub(1)).cloned().unwrap_or_default()
    }

    fn push(&self, text: String) {
        self.log
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(text);
    }
}

impl Subscriber for Capture {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, attrs: &Attributes<'_>) -> Id {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        let mut spans = self
            .spans
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        spans.push((attrs.metadata().name().to_string(), fields.rest));
        Id::from_u64(u64::try_from(spans.len()).unwrap_or(u64::MAX))
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}

    fn record_follows_from(&self, _: &Id, _: &Id) {}

    fn event(&self, event: &Event<'_>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let meta = event.metadata();
        let mut text = format!("{} {}: {}", meta.level(), meta.target(), fields.message);
        if !fields.rest.is_empty() {
            text.push(' ');
            text.push_str(&fields.rest);
        }
        self.push(text);
    }

    fn enter(&self, id: &Id) {
        let (name, fields) = self.span(id);
        if fields.is_empty() {
            self.push(format!(">{name}"));
        } else {
            self.push(format!(">{name}{{{fields}}}"));
        }
    }

    fn exit(&self, id: &Id) {
        let (name, _) = self.span(id);
        self.push(format!("<{name}"));
    }
}

/// What `body` emits, as trace tokens.
fn trace(body: impl FnOnce()) -> Vec<Tok> {
    let cap = Arc::new(Capture::default());
    tracing::subscriber::with_default(cap.clone(), body);
    let log = cap
        .log
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    log.into_iter().map(s).collect()
}

/// `trace:ShallowWater::step` (nx, ny, dt), `::add_disturbance` (nx, ny, radius) and `trace:update_sediment` (nx, ny,
/// the concentration's length, dt), on `ShallowWater::new(nx, ny, 0.1, 1.0)`.
fn shallow_rows(m: &mut Module) {
    for (nx, ny) in [(3, 3), (8, 6)] {
        for dt in [1e-3, 0.0, f64::NAN] {
            let t = trace(|| {
                if let Ok(mut sw) = ShallowWater::new(nx, ny, 0.1, 1.0) {
                    let _ = sw.step(dt);
                }
            });
            m.row(
                "trace:ShallowWater::step",
                Class::A,
                &[usz(nx), usz(ny), f(dt)],
                &t,
            );
        }
        for r in [0.3, 0.0, -1.0] {
            let t = trace(|| {
                if let Ok(mut sw) = ShallowWater::new(nx, ny, 0.1, 1.0) {
                    sw.add_disturbance(0.2, 0.2, r, 0.1);
                }
            });
            m.row(
                "trace:ShallowWater::add_disturbance",
                Class::A,
                &[usz(nx), usz(ny), f(r)],
                &t,
            );
        }
        for (len, dt) in [(nx * ny, 1e-3), (2, 1e-3), (nx * ny, -1.0)] {
            let t = trace(|| {
                if let Ok(mut sw) = ShallowWater::new(nx, ny, 0.1, 1.0) {
                    let mut c = vec![0.0; len];
                    update_sediment(&mut sw, &mut c, &SedimentConfig::default(), dt);
                }
            });
            m.row(
                "trace:update_sediment",
                Class::A,
                &[usz(nx), usz(ny), usz(len), f(dt)],
                &t,
            );
        }
    }
}

/// `trace:VofField::advect` (nx, ny, dt, the velocities' length) on `VofField::new(nx, ny, 0.1)`, still velocities.
fn vof_rows(m: &mut Module) {
    for (nx, ny) in [(4, 4), (6, 5)] {
        for (dt, len) in [(0.01, nx * ny), (0.0, nx * ny), (0.01, 3)] {
            let t = trace(|| {
                if let Ok(mut v) = VofField::new(nx, ny, 0.1) {
                    let z = vec![0.0; len];
                    let _ = v.advect(&z, &z, dt);
                }
            });
            m.row(
                "trace:VofField::advect",
                Class::A,
                &[usz(nx), usz(ny), f(dt), usz(len)],
                &t,
            );
        }
    }
}

/// `trace:PhaseField::step` (nx, ny, dt, φ₀): `PhaseField::new(nx, ny, 0.1)` with φ = φ₀ in every cell, stepped
/// without advection (φ₀ = 2 diverges at every interior cell).
fn phase_field_rows(m: &mut Module) {
    for (nx, ny) in [(4, 4), (7, 5)] {
        for (dt, phi0) in [(1e-4, 1.0), (1e-4, 2.0), (0.0, 2.0), (1e-4, 1.2)] {
            let t = trace(|| {
                if let Ok(mut p) = PhaseField::new(nx, ny, 0.1) {
                    p.phi.fill(phi0);
                    let _ = p.step(&[], &[], dt);
                }
            });
            m.row(
                "trace:PhaseField::step",
                Class::A,
                &[usz(nx), usz(ny), f(dt), f(phi0)],
                &t,
            );
        }
    }
}

/// `trace:MpmSolver::step` (n, the MPM states' count, dt) on `MpmSolver::new(8, 8, 0.1)`: n fluid particles at
/// (0.3 + 0.05 i, 0.4), mass 0.01, each state `MpmParticle::new(Fluid { 1000, 0 }, 0.0025)`.
fn mpm_rows(m: &mut Module) {
    for (n, nm, dt) in [(3, 3, 1e-3), (0, 0, 1e-3), (3, 3, 0.0), (3, 2, 1e-3)] {
        let t = trace(|| {
            if let Ok(mut s) = MpmSolver::new(8, 8, 0.1) {
                let mut ps: Vec<FluidParticle> = (0..n)
                    .map(|k| {
                        FluidParticle::new_2d(
                            0.3 + 0.05 * f64::from(u32::try_from(k).unwrap_or(0)),
                            0.4,
                            0.01,
                        )
                    })
                    .collect();
                let model = ConstitutiveModel::Fluid {
                    bulk_modulus: 1000.0,
                    viscosity: 0.0,
                };
                let mut ms: Vec<MpmParticle> =
                    (0..nm).map(|_| MpmParticle::new(model, 0.0025)).collect();
                let _ = s.step(&mut ps, &mut ms, DVec3::new(0.0, -9.81, 0.0), dt);
            }
        });
        m.row(
            "trace:MpmSolver::step",
            Class::A,
            &[usz(n), usz(nm), f(dt)],
            &t,
        );
    }
}

/// `trace:ParticleArena::free` (capacity, then ops of two integers): 0 `alloc(a)`, 1 `free` of the a-th handle
/// issued (mod the count), 2 `compact`, 4 `free` of a block of `a` particles allocated from a second arena of
/// capacity 1000 (a foreign handle).
fn arena_rows(m: &mut Module) {
    let scenarios: [(usize, &[(i64, i64)]); 4] = [
        (20, &[(0, 5), (0, 5), (1, 0), (1, 0)]),
        (20, &[(0, 5), (0, 5), (2, 0), (1, 1)]),
        (20, &[(0, 5), (4, 50), (4, 2)]),
        (
            30,
            &[
                (0, 10),
                (0, 10),
                (1, 1),
                (1, 0),
                (1, 1),
                (2, 0),
                (0, 4),
                (1, 2),
            ],
        ),
    ];
    for (cap, ops) in scenarios {
        let t = trace(|| {
            let mut a = ParticleArena::new(cap);
            let mut b = ParticleArena::new(1000);
            let mut issued = Vec::new();
            for &(code, arg) in ops {
                let n = usize::try_from(arg).unwrap_or(0);
                match code {
                    0 => {
                        if let Some(h) = a.alloc(n) {
                            issued.push(h);
                        }
                    }
                    1 => {
                        if !issued.is_empty() {
                            a.free(issued[n % issued.len()]);
                        }
                    }
                    2 => {
                        a.compact();
                    }
                    _ => {
                        if let Some(h) = b.alloc(n) {
                            a.free(h);
                        }
                    }
                }
            }
        });
        let mut ins = vec![usz(cap)];
        for &(code, arg) in ops {
            ins.extend([i(code), i(arg)]);
        }
        m.row("trace:ParticleArena::free", Class::A, &ins, &t);
    }
}

/// n particles of mass 0.01 at (0.07 k, 0.01 k, 0).
#[allow(clippy::cast_precision_loss)]
fn line(n: usize) -> Vec<FluidParticle> {
    (0..n)
        .map(|k| FluidParticle::new(DVec3::new(0.07 * k as f64, 0.01 * k as f64, 0.0), 0.01))
        .collect()
}

/// `trace:couple_sph_bodies` (n, bodies): `line(n)` and spheres of radius 0.2 and mass 1 at (0.1 j, 0, 0), h 0.1,
/// stiffness 100, damping 1. `trace:drag_from_particles` and `trace:drag_from_particles_indexed` (n): a sphere of
/// radius 0.2 at the origin, ρ 1000, Cd 0.47, sample radius 0.5 (the indexed one through a hash of cell 0.25 holding
/// `line(n)`). `trace:particle_level_set` (nx, ny, the level set's length, n) at dx 0.1, radius 0.05.
/// `trace:FlipSolver::step` (n, dt, APIC 0 / 1) on `FlipSolver::new(8, 8, 0.1, 0.95)` with n particles at
/// (0.2 + 0.03 k, 0.3).
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn coupling_rows(m: &mut Module) {
    let sphere = |x: f64, mass: f64| {
        RigidBody::new(
            DVec3::new(x, 0.0, 0.0),
            mass,
            BodyShape::Sphere { radius: 0.2 },
        )
    };
    for (n, nb) in [(5, 1), (0, 2), (6, 0)] {
        let t = trace(|| {
            let mut ps = line(n);
            let mut bs: Vec<RigidBody> = (0..nb).map(|j| sphere(0.1 * j as f64, 1.0)).collect();
            couple_sph_bodies(&mut ps, &mut bs, 0.1, 100.0, 1.0);
        });
        m.row("trace:couple_sph_bodies", Class::A, &[usz(n), usz(nb)], &t);
    }
    for n in [4, 0] {
        let t = trace(|| {
            let _ = drag_from_particles(&sphere(0.0, 1.0), &line(n), 1000.0, 0.47, 0.5);
        });
        m.row("trace:drag_from_particles", Class::A, &[usz(n)], &t);
        let t = trace(|| {
            let ps = line(n);
            let mut hash = SpatialHash::new(0.25).expect("a positive cell size");
            for (k, p) in ps.iter().enumerate() {
                hash.insert(
                    Vec3::new(
                        p.position.x as f32,
                        p.position.y as f32,
                        p.position.z as f32,
                    ),
                    k,
                );
            }
            let _ = drag_from_particles_indexed(&sphere(0.0, 1.0), &ps, &hash, 1000.0, 0.47, 0.5);
        });
        m.row("trace:drag_from_particles_indexed", Class::A, &[usz(n)], &t);
    }
    for (nx, ny, len, n) in [(4, 3, 12, 3), (4, 3, 11, 3), (0, 5, 0, 2), (3, 3, 9, 0)] {
        let t = trace(|| {
            let mut ls = vec![0.0; len];
            particle_level_set(&mut ls, nx, ny, 0.1, &line(n), 0.05);
        });
        m.row(
            "trace:particle_level_set",
            Class::A,
            &[usz(nx), usz(ny), usz(len), usz(n)],
            &t,
        );
    }
    for (n, dt, apic) in [
        (4, 1e-3, false),
        (4, 1e-3, true),
        (0, 1e-3, false),
        (4, 0.0, false),
    ] {
        let t = trace(|| {
            if let Ok(mut s) = FlipSolver::new(8, 8, 0.1, 0.95) {
                s.use_apic = apic;
                let mut ps: Vec<FluidParticle> = (0..n)
                    .map(|k| FluidParticle::new_2d(0.2 + 0.03 * k as f64, 0.3, 0.01))
                    .collect();
                let _ = s.step(&mut ps, DVec3::new(0.0, -9.81, 0.0), dt);
            }
        });
        m.row(
            "trace:FlipSolver::step",
            Class::A,
            &[usz(n), f(dt), i(i64::from(apic))],
            &t,
        );
    }
}

/// `trace:FluidGrid::step` (nx, ny, boundary 0 / 1 / 2, flags, viscosity, vorticity, dt) on `FluidGrid::new(nx, ny,
/// 0.1)` at rest under `GridConfig::smoke()` with those set; flags 1 MacCormack, 2 BFECC, 4 multigrid, 8 CG, 16
/// Smagorinsky (Cs 0.1). `trace:FluidGrid::diffuse` (nx, ny, the field's length, iterations): a zero field, rate
/// 0.01, dt 0.1, dx 0.1. `trace:KEpsilon::step` (nx, ny, the velocities' length, dt) on `KEpsilon::new(nx, ny)`, dx
/// 0.1, still flow. `trace:MacGrid::step` (nx, ny, dt) on `MacGrid::new(nx, ny, 0.1)` under smoke. `trace:
/// apply_ghost_fluid` (nx, ny, the pressure's and the level set's lengths, dx), zeros, σ 0.07.
fn grid_rows(m: &mut Module) {
    let steps: [(usize, usize, i64, i64, f64, f64, f64); 12] = [
        (8, 8, 0, 0, 0.0, 0.0, 0.01),
        (8, 8, 0, 0, 0.001, 0.0, 0.01),
        (8, 8, 1, 1, 0.001, 0.5, 0.01),
        (8, 8, 0, 2, 0.0, 0.0, 0.01),
        (8, 8, 0, 4, 0.0, 0.0, 0.01),
        (8, 8, 0, 8, 0.0, 0.0, 0.01),
        (8, 8, 0, 12, 0.0, 0.0, 0.01),
        (8, 8, 2, 0, 0.001, 0.0, 0.01),
        (8, 8, 2, 1, 0.0, 0.0, 0.01),
        (8, 8, 0, 16, 0.0, 0.0, 0.01),
        (4, 4, 0, 0, 0.0, 0.5, 0.01),
        (8, 8, 0, 0, 0.0, 0.0, 0.0),
    ];
    for (nx, ny, b, flags, visc, vort, dt) in steps {
        let t = trace(|| {
            if let Ok(mut g) = FluidGrid::new(nx, ny, 0.1) {
                let mut c = GridConfig::smoke();
                c.dt = dt;
                c.viscosity = visc;
                c.vorticity_confinement = vort;
                c.boundary = match b {
                    0 => BoundaryCondition::NoSlip,
                    1 => BoundaryCondition::FreeSlip,
                    _ => BoundaryCondition::Periodic,
                };
                c.use_maccormack = flags & 1 != 0;
                c.use_bfecc = flags & 2 != 0;
                c.use_multigrid = flags & 4 != 0;
                c.use_cg = flags & 8 != 0;
                c.smagorinsky_cs = if flags & 16 != 0 { 0.1 } else { 0.0 };
                let _ = g.step(&c);
            }
        });
        m.row(
            "trace:FluidGrid::step",
            Class::A,
            &[usz(nx), usz(ny), i(b), i(flags), f(visc), f(vort), f(dt)],
            &t,
        );
    }
    for (nx, ny, len, it) in [
        (8, 8, 64, 10),
        (8, 8, 63, 10),
        (2, 8, 16, 10),
        (8, 8, 64, 0),
    ] {
        let t = trace(|| {
            let mut field = vec![0.0; len];
            FluidGrid::diffuse(&mut field, nx, ny, 0.01, 0.1, 0.1, it);
        });
        m.row(
            "trace:FluidGrid::diffuse",
            Class::A,
            &[usz(nx), usz(ny), usz(len), usz(it)],
            &t,
        );
    }
    for (nx, ny, len, dt) in [
        (6, 6, 36, 0.01),
        (6, 6, 35, 0.01),
        (6, 6, 36, 0.0),
        (2, 2, 4, 0.01),
    ] {
        let t = trace(|| {
            let mut k = KEpsilon::new(nx, ny);
            let z = vec![0.0; len];
            k.step(&z, &z, nx, ny, 0.1, dt);
        });
        m.row(
            "trace:KEpsilon::step",
            Class::A,
            &[usz(nx), usz(ny), usz(len), f(dt)],
            &t,
        );
    }
    for (nx, ny, dt) in [(8, 8, 0.01), (6, 7, 0.01), (8, 8, 0.0)] {
        let t = trace(|| {
            if let Ok(mut g) = MacGrid::new(nx, ny, 0.1) {
                let mut c = GridConfig::smoke();
                c.dt = dt;
                let _ = g.step(&c);
            }
        });
        m.row(
            "trace:MacGrid::step",
            Class::A,
            &[usz(nx), usz(ny), f(dt)],
            &t,
        );
    }
    for (nx, ny, pl, ll, dx) in [
        (6, 6, 36, 36, 0.1),
        (6, 6, 35, 36, 0.1),
        (6, 6, 36, 36, 0.0),
    ] {
        let t = trace(|| {
            let mut p = vec![0.0; pl];
            let ls = vec![0.0; ll];
            apply_ghost_fluid(&mut p, &ls, nx, ny, dx, 0.07);
        });
        m.row(
            "trace:apply_ghost_fluid",
            Class::A,
            &[usz(nx), usz(ny), usz(pl), usz(ll), f(dx)],
            &t,
        );
    }
}

/// side × side particles of mass 0.01, 0.01 apart from the origin, with temperature 600 and fuel 1 (so combustion
/// reacts), and velocity (0.1·(k mod 3), 0).
#[allow(clippy::cast_precision_loss)]
fn block(side: usize) -> Vec<FluidParticle> {
    (0..side * side)
        .map(|k| {
            let mut p =
                FluidParticle::new_2d(0.01 * (k % side) as f64, 0.01 * (k / side) as f64, 0.01);
            p.velocity = DVec3::new(0.1 * (k % 3) as f64, 0.0, 0.0);
            p.temperature = 600.0;
            p.fuel = 1.0;
            p
        })
        .collect()
}

/// CSR lists of every pair within h (self included), or a broken form: kind 1 one offset short, kind 2 an index n.
#[allow(clippy::cast_possible_truncation)]
fn lists(ps: &[FluidParticle], h: f64, kind: i64) -> (Vec<u32>, Vec<usize>) {
    let mut off = vec![0u32];
    let mut idx = Vec::new();
    for a in ps {
        for (j, b) in ps.iter().enumerate() {
            if a.position.distance(b.position) < h {
                idx.push(j);
            }
        }
        off.push(idx.len() as u32);
    }
    match kind {
        1 => {
            off.pop();
        }
        2 => idx.push(ps.len()),
        _ => {}
    }
    if kind == 2
        && let Some(last) = off.last_mut()
    {
        *last += 1;
    }
    (off, idx)
}

/// The SPH scenarios, on `block(side)` under `FluidConfig::water_2d()` (h 0.05) unless said:
/// - `trace:SphSolver::step` (side, kind) on `SphSolver::new()`, viscosity 0.001 (kind 1: dt 0; kind 2: the first
///   particle's vx NaN, which diverges); `trace:step` the free step, the same; `::step_pcisph` and `::step_dfsph`
///   (side, max iterations), error 0.01; `::step_multiphase` (side, phases 1 / 2, k, phase, NaN 0 / 1):
///   `MultiPhaseConfig::single(1000, 2000, 0.001)` (twice for 2), the first k particles in `phase`, the first one's
///   vx NaN for NaN 1;
/// - `trace:pressure_force` and `trace:viscosity_force` (side, index), viscosity 0.001;
/// - the list functions (side, lists kind, zero): `lists(block, 0.05, kind)`; zero 1 makes their first parameter 0
///   (ve's modulus, heat's diffusivity, δ, μ, h), dt 0.001: `update_viscoelastic` (modulus 1000, τ 0.1),
///   `update_heat` (diffusivity 0.1), `apply_delta_sph` (c 10, δ 0.1), `apply_implicit_viscosity` (μ 0.001, 2
///   iterations), `compute_gradient_corrections` (h 0.05);
/// - `trace:update_phase_change` (side), `PhaseChangeConfig::water_ice(0, 1)`; `trace:update_combustion` and
///   `trace:update_reaction` (side), `CombustionConfig::default()` (the provider too), dt 0.001; `trace:apply_contact_angle` (side, h 0 / 1), angle 1, σ 0.07;
///   `trace:split_particle` (children), offset 0.01; `trace:merge_particles` (), of `block(2)`'s first two;
///   `trace:sort_by_zorder` (side, cell size); `trace:generate_secondary_particles` (side), threshold 0, lifetime 1.
#[allow(clippy::too_many_lines)]
fn sph_rows(m: &mut Module) {
    let water = FluidConfig::water_2d();
    for (side, kind) in [(3, 0), (3, 1), (0, 0), (3, 2)] {
        let mut c = water;
        if kind == 1 {
            c.dt = 0.0;
        }
        let start = |side: usize| {
            let mut ps = block(side);
            if kind == 2 {
                ps[0].velocity.x = f64::NAN;
            }
            ps
        };
        let t = trace(|| {
            let mut ps = start(side);
            let _ = SphSolver::new().step(&mut ps, &c, 0.001);
        });
        m.row("trace:SphSolver::step", Class::A, &[usz(side), i(kind)], &t);
        let t = trace(|| {
            let mut ps = start(side);
            let _ = pravash::sph::step(&mut ps, &c, 0.001);
        });
        m.row("trace:step", Class::A, &[usz(side), i(kind)], &t);
    }
    for (side, it) in [(3, 5), (3, 1), (0, 5)] {
        let t = trace(|| {
            let mut ps = block(side);
            let _ = SphSolver::new().step_pcisph(&mut ps, &water, 0.001, it, 0.01);
        });
        m.row(
            "trace:SphSolver::step_pcisph",
            Class::A,
            &[usz(side), usz(it)],
            &t,
        );
        let t = trace(|| {
            let mut ps = block(side);
            let _ = SphSolver::new().step_dfsph(&mut ps, &water, 0.001, it, 0.01);
        });
        m.row(
            "trace:SphSolver::step_dfsph",
            Class::A,
            &[usz(side), usz(it)],
            &t,
        );
    }
    for (side, phases, k, phase, nan) in [
        (3, 1, 0, 5, 0),
        (3, 2, 2, 5, 0),
        (3, 1, 4, 5, 0),
        (0, 1, 0, 5, 0),
        (3, 1, 2, 1, 0),
        (3, 2, 2, 2, 0),
        (3, 2, 2, 1, 0),
        (3, 1, 0, 5, 1),
    ] {
        let t = trace(|| {
            let mut ps = block(side);
            for p in ps.iter_mut().take(k) {
                p.phase = phase;
            }
            if nan == 1 {
                ps[0].velocity.x = f64::NAN;
            }
            let mut mp = MultiPhaseConfig::single(1000.0, 2000.0, 0.001);
            if phases == 2 {
                let first = mp.phases[0];
                mp.phases.push(first);
            }
            let _ = SphSolver::new().step_multiphase(&mut ps, &water, &mp);
        });
        m.row(
            "trace:SphSolver::step_multiphase",
            Class::A,
            &[usz(side), usz(phases), usz(k), i(i64::from(phase)), i(nan)],
            &t,
        );
    }
    for (side, idx) in [(3, 4), (2, 0)] {
        let t = trace(|| {
            let _ = pressure_force(idx, &block(side), 0.05);
        });
        m.row("trace:pressure_force", Class::A, &[usz(side), usz(idx)], &t);
        let t = trace(|| {
            let _ = viscosity_force(idx, &block(side), 0.05, 0.001);
        });
        m.row(
            "trace:viscosity_force",
            Class::A,
            &[usz(side), usz(idx)],
            &t,
        );
    }
    for (side, kind, zero) in [(3, 0, 0), (3, 1, 0), (3, 2, 0), (3, 1, 1), (0, 1, 0)] {
        let z = |x: f64| if zero == 1 { 0.0 } else { x };
        let ins = [usz(side), i(kind), i(zero)];
        let t = trace(|| {
            let mut ps = block(side);
            let (o, x) = lists(&ps, 0.05, kind);
            update_viscoelastic(
                &mut ps,
                &o,
                &x,
                0.05,
                &ViscoelasticConfig::new(z(1000.0), 0.1),
                0.001,
            );
        });
        m.row("trace:update_viscoelastic", Class::A, &ins, &t);
        let t = trace(|| {
            let mut ps = block(side);
            let (o, x) = lists(&ps, 0.05, kind);
            update_heat(&mut ps, &o, &x, 0.05, &HeatConfig::new(z(0.1)), 0.001);
        });
        m.row("trace:update_heat", Class::A, &ins, &t);
        let t = trace(|| {
            let mut ps = block(side);
            let (o, x) = lists(&ps, 0.05, kind);
            apply_delta_sph(&mut ps, &o, &x, 0.05, 10.0, z(0.1), 0.001);
        });
        m.row("trace:apply_delta_sph", Class::A, &ins, &t);
        let t = trace(|| {
            let mut ps = block(side);
            let (o, x) = lists(&ps, 0.05, kind);
            apply_implicit_viscosity(&mut ps, &o, &x, 0.05, z(0.001), 0.001, 2);
        });
        m.row("trace:apply_implicit_viscosity", Class::A, &ins, &t);
        let t = trace(|| {
            let ps = block(side);
            let (o, x) = lists(&ps, 0.05, kind);
            let _ = compute_gradient_corrections(&ps, &o, &x, z(0.05));
        });
        m.row("trace:compute_gradient_corrections", Class::A, &ins, &t);
    }
    for side in [3, 0] {
        let t = trace(|| {
            update_phase_change(&mut block(side), &PhaseChangeConfig::water_ice(0, 1), 0.001)
        });
        m.row("trace:update_phase_change", Class::A, &[usz(side)], &t);
        let t = trace(|| update_combustion(&mut block(side), &CombustionConfig::default(), 0.001));
        m.row("trace:update_combustion", Class::A, &[usz(side)], &t);
        let t = trace(|| update_reaction(&mut block(side), &CombustionConfig::default(), 0.001));
        m.row("trace:update_reaction", Class::A, &[usz(side)], &t);
        let t = trace(|| {
            let _ = generate_secondary_particles(&block(side), 0.0, 1.0);
        });
        m.row(
            "trace:generate_secondary_particles",
            Class::A,
            &[usz(side)],
            &t,
        );
        for h in [0, 1] {
            let mut c = water;
            if h == 0 {
                c.smoothing_radius = 0.0;
            }
            let t = trace(|| apply_contact_angle(&mut block(side), &c, 1.0, 0.07));
            m.row(
                "trace:apply_contact_angle",
                Class::A,
                &[usz(side), i(h)],
                &t,
            );
        }
        for cell in [0.02, 0.0] {
            let t = trace(|| sort_by_zorder(&mut block(side), cell));
            m.row("trace:sort_by_zorder", Class::A, &[usz(side), f(cell)], &t);
        }
    }
    for n in [0, 1, 4] {
        let t = trace(|| {
            let _ = split_particle(&block(2)[0], n, 0.01);
        });
        m.row("trace:split_particle", Class::A, &[usz(n)], &t);
    }
    let t = trace(|| {
        let b = block(2);
        let _ = merge_particles(&b[0], &b[1]);
    });
    m.row("trace:merge_particles", Class::A, &[], &t);
}

pub fn emit() -> Module {
    let mut m = Module::new("logging");
    arena_rows(&mut m);
    coupling_rows(&mut m);
    grid_rows(&mut m);
    mpm_rows(&mut m);
    phase_field_rows(&mut m);
    shallow_rows(&mut m);
    sph_rows(&mut m);
    vof_rows(&mut m);
    let mut row = |env: Option<&str>, default: &str| {
        let ins = vec![env.map_or(Tok::None, s), s(default)];
        let out: Vec<Tok> = levels(filter(env, default)).into_iter().map(i).collect();
        m.row("init_with_level", Class::A, &ins, &out);
    };
    for d in DEFAULTS {
        row(None, d);
    }
    for e in ENVS {
        row(Some(e), "info");
    }
    for e in ["", "bogus!", "pravash=debug", "[sph::step]=trace"] {
        for d in &DEFAULTS[1..] {
            row(Some(e), d);
        }
    }
    m
}
