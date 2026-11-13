"""The expected outputs of the ported examples (docs/examples/, M8 part c), recomputed without the port.

Usage, from the repo root (local only, as validate.py, whose transcriptions it runs on the recorded glibc):
    python3 -I -B scripts/oracle/examples.py .            # check docs/examples/*.out
    python3 -I -B scripts/oracle/examples.py . --write    # write them

Rust 1.3.0's three examples are ported as docs/examples/<name>.cyr, and docs/examples/<name>.out is what each must
print; scripts/doc_examples.py runs the programs and compares. This script derives the .out files from the oracle:
- shallow_water.out is exactly what `cargo run --example shallow_water` prints;
- basic_fluid.out and multiphase.out are what the Rust example prints, with each frame line recomputed by
  validate.py's transcription of the solver on the port's f64 neighbour keys. The same transcription on Rust's f32
  keys must give exactly the Rust example's lines first, or the check fails. The two separate at the second step:
  the f32 keys move a particle just below a cell boundary (x = 0.3999999999999997) into the next cell, so 49
  particles sum the same neighbours in another order, and this scene, at thousands of m/s, grows the last-bit
  difference (ADR 0002 #2).
"""
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path

root = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), ".")).resolve()
WRITE = "--write" in sys.argv[1:]

spec = importlib.util.spec_from_file_location("validate", root / "scripts/oracle/validate.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

# FluidConfig::water_2d in the corpus's field order: dt, gravity, smoothing_radius, gas_constant, rest_density,
# bounds_min, bounds_max, boundary_damping.
WATER_2D = [0.001, 0.0, -9.81, 0.0, 0.05, 2000.0, 1000.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.5]
VISCOSITY = 0.001  # FluidMaterial::WATER.viscosity


def block(ox, oy, sx, sy, spacing, mass):
    """`create_particle_block`: FluidParticle::new_2d rows in the corpus's 18-value form."""
    nx, ny = math.ceil(sx / spacing), math.ceil(sy / spacing)
    return [[ox + ix * spacing, oy + iy * spacing, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mass, 0,
             1.0, 0.0, 1.0, 293.15, 0.0] for iy in range(ny) for ix in range(nx)]


def max_speed(ps):
    """Rust's `max_speed`: the square root of the largest squared speed."""
    m = 0.0
    for p in ps:
        m = v.r_max(m, v.sum_sq(p[3], p[4], p[5]))
    return math.sqrt(m)


def basic_fluid():
    ps, prev, out = block(0.2, 0.5, 0.3, 0.3, 0.02, 0.001), None, []
    for frame in range(100):
        err, prev = v.r_solver_step(ps, WATER_2D, VISCOSITY, 0.0, False, prev)
        if err:
            return [f"Err {err}"]
        if frame % 10 == 0:
            ke = 0.0
            for p in ps:
                ke += (0.5 * p[11]) * v.sum_sq(p[3], p[4], p[5])
            out.append(f"Frame {frame:>4}  KE: {ke:>10.4f}  max_speed: {max_speed(ps):>8.4f}")
    return out


def multiphase():
    oil = block(0.2, 0.3, 0.3, 0.2, 0.02, 0.0008)
    for p in oil:
        p[12] = 1
    ps, out = block(0.2, 0.1, 0.3, 0.2, 0.02, 0.001) + oil, []
    phases = [(1000.0, 2000.0, 0.001), (800.0, 1500.0, 0.03)]
    for frame in range(100):
        err = v.r_multiphase_step(ps, WATER_2D, 0.0, 0.05, phases)
        if err:
            return [f"Err {err}"]
        if frame % 10 == 0:
            water = sum(1 for p in ps if p[12] == 0)
            oil_n = sum(1 for p in ps if p[12] == 1)
            out.append(f"Frame {frame:>4}  water: {water}  oil: {oil_n}  max_speed: {max_speed(ps):>8.4f}")
    return out


def rust_example(name):
    r = subprocess.run(["cargo", "run", "--release", "--locked", "--quiet", "--manifest-path",
                        str(root / "rust-old/Cargo.toml"), "--example", name, "--features", "sph shallow"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"cargo run --example {name} failed: {r.stderr.strip()[-300:]}")
    return r.stdout


def main():
    fails = []
    for name, run in (("shallow_water", None), ("basic_fluid", basic_fluid), ("multiphase", multiphase)):
        rust = rust_example(name)
        want = rust
        if run is not None:
            rust_frames = [ln for ln in rust.split("\n") if ln.startswith("Frame ")]
            v.NB64 = False
            if run() != rust_frames:
                fails.append(f"{name}: the transcription on f32 keys doesn't reproduce the Rust example")
                continue
            v.NB64 = True
            port = iter(run())
            want = re.sub(r"^Frame .*$", lambda m: next(port), rust, flags=re.M)
        path = root / "docs/examples" / f"{name}.out"
        if WRITE:
            path.write_text(want)
            print(f"{path.relative_to(root)}: written")
        elif not path.exists() or path.read_text() != want:
            fails.append(f"{path.relative_to(root)} is stale or missing (run with --write)")
        else:
            same = "Rust's output" if want == rust else "Rust's output with the f64-key frames"
            print(f"{path.relative_to(root)}: current ({same})")
    for f in fails:
        print("EXAMPLE", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
