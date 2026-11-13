# Examples

Rust 1.3.0's three examples (`rust-old/examples/`), ported. Each is a whole consumer program: it includes
`lib/pravash.cyr`, where `cyrius deps` vendors the bundle. They print through the stdlib's `fmt`, which a consumer
lists in its own `[deps] stdlib` (the bundle doesn't need it).

| Example | What it runs | Output |
|---|---|---|
| [`basic_fluid.cyr`](basic_fluid.cyr) | An SPH dam break: 225 particles, 100 WCSPH steps, the kinetic energy and the fastest particle every 10 frames | [`basic_fluid.out`](basic_fluid.out) |
| [`shallow_water.cyr`](shallow_water.cyr) | A splash in a 40 × 40 shallow-water basin with Manning friction: 200 steps, the volume and the highest wave every 20 frames | [`shallow_water.out`](shallow_water.out) |
| [`multiphase.cyr`](multiphase.cyr) | Oil above water: 300 particles, 100 multiphase steps with interface tension | [`multiphase.out`](multiphase.out) |

To run one in this repository, point its include at the bundle and build it:

```sh
cyrius distlib
mkdir -p build
sed 's#lib/pravash.cyr#dist/pravash.cyr#' docs/examples/basic_fluid.cyr > build/basic_fluid.cyr
cyrius build build/basic_fluid.cyr build/basic_fluid && ./build/basic_fluid
```

`python3 -I -B scripts/doc_examples.py .` builds and runs all three and checks that each prints exactly its `.out`
file.

## Where the outputs come from

The `.out` files are derived from the oracle by `scripts/oracle/examples.py` (local only, like `validate.py`):

- **`shallow_water.out`** is exactly what Rust 1.3.0's example prints.
- **`basic_fluid.out` and `multiphase.out`** are Rust's output with each frame line recomputed by `validate.py`'s
  Python transcription of the solver, on the f64 neighbour keys 2.0 uses
  ([ADR 0002 #2](../adr/0002-port-api-divergences.md)). Frame 0 is Rust's line; from frame 10 the numbers differ.
  On Rust's f32 keys, the same transcription prints Rust's own lines exactly, so the keys are the only difference.

  At the second step of `basic_fluid`, the f32 keys put a particle at x = 0.3999999999999997 one cell over. 49
  particles then sum the same neighbours in another order. The scene reaches 6,800 m/s by frame 10, so the last-bit
  difference grows into different numbers. The shallow-water solver has no neighbour grid, and it prints Rust's
  numbers.
