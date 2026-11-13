# Changelog

## [Unreleased]

### Added (Cyrius)
- **coupling** (M6 part a): `src/coupling.cyr`, the rigid bodies of `coupling.rs`. The FLIP solver follows in part b.
  ADR 0002 #23.
  - **`BodyShape`** is the kind enum `PvBodyShape` and the payload struct `PvShape` (`PvShape_sphere(r)`,
    `PvShape_box(hx, hy, hz)`). Its methods are `body_shape_signed_distance`, `_volume`, `_bounding_radius` and
    `_projected_area`.
  - **`PvRigidBody`** has `PvRigidBody_new`, `b.signed_distance(point)` and `b.surface_normal(point)`.
  - **The free functions:** `couple_sph_bodies`, `integrate_bodies`, `effective_mass`,
    `integrate_bodies_with_added_mass`, `drag_from_particles` and `particle_level_set`. The level set returns 1, or 0
    when it leaves the buffer unchanged.
  - **`drag_from_particles_indexed`** takes the caller's cells-only neighbour grid, built by the new
    `PvNeighborGrid_build(g, ps, n, cell_size)`, where Rust takes a hisab `SpatialHash` of f32 positions.
    - The grid's keys are the f64 keys (ADR 0002 #2), and it takes the cell sizes hisab 3.4.1 takes.
    - The query visits the occupied cells of Rust's block in Rust's (x, y, z) order, so any radius returns. Rust's
      walks every cell of the block, so an infinite radius never returns.
    - The grid gains an eighth column (224 bytes; the solver 496).
  - **JSON:** hand codecs for `BodyShape`, externally tagged with struct-variant fields, and `RigidBody`
    (`src/json.cyr`).
  - **Parity:** a new golden module, `tests/golden/coupling.tsv`: 4,798 rows from the generator's
    `m_coupling.rs`.
    - Every row but one is bit-exact on the first replay.
    - The exception is an indexed drag about a body at 10⁶, where the f32 hash orders the particles differently. It
      is mapped to `coupling.d3.tsv` from validate.py's f64 transcription and *accepted (ADR 2)*.
    - validate.py recomputes every row, including the indexed drag on transcriptions of both hashes.
  - **Tests:** `tests/coupling.tcyr` ports 22 of `coupling.rs`'s 33 Rust tests (the FLIP solver's 11 come with part
    b), plus Cyrius-only groups: 55 checks. `tests/integration.tcyr` adds the two coupling integration tests (31 of
    33). Rust's `coupling_sphere_falls_through_fluid` couples a clone of the body, so the port couples a copy.
    - The record layouts and the grid's cell-size rule.
    - The block walk against a brute-force enumeration on 40 blocks.
    - The indexed drag against the brute force at the origin, 10⁶ and 10¹², and with an infinite radius.
    - The level set's refusals.
  - **Mutations:** 27 of 31 port mutations fail rows on the first run, a 32nd (a misnamed `Box` key) fails 47, and
    all 8 of validate.py's are caught.
    - Two survivors were corpus gaps, closed by rows that draw no random numbers: a walk that skipped a column, and a
      level set longer than nx·ny.
    - One is equivalent: a zero-cell block is only reached when no particle can pass the filter.
    - Rust's grouping of the sphere volume, changed, moves 25 rows by 1 ulp. That is inside class B's bound, so it
      shows in the histogram only.

### Performance
- **coupling** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M6 part a; pinned to core 15,
  against Rust's criterion groups and a twin):

  | Row | Cyrius / Rust |
  |---|---|
  | `couple_sph_bodies`, 625 particles | 5.5–5.8× |
  | `drag_from_particles`, 625 | 7.9× |
  | `particle_level_set`, all three sizes | 4.2–4.3× |
  | `drag_from_particles_indexed`, 625 | 0.70–0.72× |
  | `drag_from_particles_indexed`, 10,000 | 2.2–2.3× |
  | the grid build | 3.4× (625), 5.1× (10,000) |

  - **Where the speed came from:** the distance loops now read the shape from a per-body context through a
    two-argument call, and the coupling runs its bodies outside its particles. The bits are the same.
  - **A/B against the first version:** the drag 1.53× faster, the indexed drag 1.29×, the coupling 1.27×.
  - **Why the indexed drag can beat Rust's:** Rust's query walks every cell of a 3D block, even for 2D particles.

### Added (Cyrius)
- **sph** (M5 part e): the solver's thread count, and the D3 switch to f64 neighbour keys. M5 is complete.
  - **`s.set_threads(n)`** (ADR 0002 #4, amended): the solver's pool runs Rust's three `parallel` loops, `step`'s
    densities and accelerations and `step_multiphase`'s accelerations, as `_pv_par_for` kernels. Each particle is
    written by one thread, so every count gives the serial bits. Drop stops the pool and keeps the count. PCISPH and
    DFSPH have no `parallel` loops in Rust and stay serial.
    - **Parity:** every solver step row (147) replays at 2, 3, 4 and 8 threads: 0 differ from one thread.
    - **Tests:** `tests/par.tcyr` adds the solver's thread-count lifecycle (the pool starts on the first step, 20 more
      steps at 4 threads take no heap) and Euler, Verlet and multiphase at 2–16 threads against one: 50 checks.
      Mutants that route a sum through a shared global are caught by both the replay and the test.
  - **f64 neighbour keys** (port plan D3, ADR 0002 #2, amended):
    - each point's cell is floor(x · fl(1/h)) as a saturating i64, hisab 3.4.1's `_sh_cell_coord`;
    - a particle's list is the 27 cells at its own key ± 1 per axis, in integers.

    The switch drops the f32 probe points, which at large coordinates land in other cells and list a cell twice, and
    the f32 grid's cell-size reuse band. The grid loses its f32 position column (200 bytes; the solver 472).
    - **Parity:** the rows the switch changes are mapped by name to two new tables, which validate.py writes from its
      own transcription of the f64 grid: `tests/golden/math.d3.tsv` (558 of the 2,460 cell-key rows) and
      `sph.d3.tsv` (all 30 list rows, 6 `step`, 1 `step_dfsph`, 1 `step_multiphase`). All 596 pass, *accepted
      (ADR 2)*; the other rows keep the oracle's outputs.
    - **Tests:** lattices at x = 10⁶ and 10¹², where the f32 lists drifted, still list every particle within h once.
      Keys saturated at ±∞ on each axis never list each other, and a NaN coordinate is cell 0 (no row can hold a ±∞
      position: the oracle panics on `cx + 1`). 190 checks.
- **The parity harness**: `tests/golden/<module>.d3.tsv` loads into the corrected-oracle table, each row tagged with its
  ADR, and reports *accepted (ADR 2)* apart from ADR 4. Thread-count replays share one helper, `_par_threads_agree`.
  validate.py writes and checks the `.d3.tsv` files (`--write`, as the `.cr.tsv` ones), and the generator's `--check`
  counts each as its module's.

### Fixed
- **The parity harness**: an unusable solver row (-1, a phase column that can't grow) fell through `_par_sph_d` and
  was read as pending. It is now reported as unusable, which fails the module.

### Performance
- **f64 keys** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M5 part e; A/B against the f32
  grid): the 1,000-particle neighbour build 1.5–1.6× faster in 2D (now 0.93× Rust's) and 1.4–1.6× in 3D (1.25–1.34×
  Rust's, from 1.9×); the dense 625-particle step 12–14% faster.
- **Thread counts** (pinned to cores 8–15, on the f64 keys): at 8 threads the dense 625-particle step runs 2.1–2.3×
  faster than at one, the 2,500-particle one 3.1–3.2×, and the multiphase step 1.8× (its density loop is serial, as
  Rust's). Rust's rayon build gains at most 1.1×, 1.7× and 1.2× on the same rows, so at 8 threads the port is 1.2–1.45×,
  2.7–2.9× and 1.8–1.9× Rust's time (2.8–3.0×, 5.2× and 2.8–3.0× at one).

### Added (Cyrius)
- **sph** (M5 part d): `src/sph_multiphase.cyr`, the solver's multiphase step.
  - **`PvPhaseProperties`** (rest density, gas constant, viscosity) and **`PvMultiPhaseConfig`**. Its phases are a
    column of three words a phase, with `phase_count()`, `push_phase(p)` and `PvMultiPhaseConfig_drop`. The
    constructors `PvMultiPhaseConfig_single` and `_two_phase` return it by value, with no phase when the column can't
    be allocated (Rust aborts there), which the step then refuses.
  - **`s.step_multiphase(ps, n, config, phase_config)`** in Rust's order:
    - the checks (the config, n = 0, no phase, the f32 h);
    - the densities, each pressure by its own phase's equation of state;
    - the forces from a snapshot: viscosity at the mean of the two phases', the free-surface CSF term for σ > 0, and
      a second CSF term on the phase indicator between unlike phases;
    - the divergence check, symplectic Euler and the walls.

    A phase outside the list reads phase 0. Rust also warns once a step; that waits for M7's logging. ADR 0002 #22.
  - **Parity:** 26 new class S rows, 1,534 in all. They cover the plan's two-phase layer to 300 steps; each tension
    alone, both, and 0, negative and NaN; three phases with indices outside the list; Rust's test shapes; a random
    splash; and the errors, among them the empty list with and without particles.
    - All 25 rows of the first set passed on the first replay. `validate.py` transcribes the step: 49,968 rows,
      0 failures.
    - The mutation tests found one gap, the 1e-10 density floor. The 26th row closes it: masses of 1e-15, so every
      density sits under the floor. It draws no random numbers.
  - **Tests:** the five multiphase Rust tests, which makes all 79 of `sph.rs`.

### Performance
- **Multiphase** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M5 part d; Rust's criterion
  suite has no multiphase group, so against a twin). On the dense 625-particle block it is 3.4–3.5× with one phase,
  3.5× with two phases and interface tension, and 3.8–3.9× with free-surface tension added. That is part a's WCSPH
  step's ratio (3.2–3.6×).

### Added (Cyrius)
- **sph** (M5 part c): `src/sph_dfsph.cyr`, the solver's DFSPH step.
  - **`s.step_dfsph(ps, n, config, viscosity, max_iterations, max_error)`** in Bender & Koschier's order:
    - the densities and α, the non-pressure velocity, the density solver;
    - the advection, a second neighbour build and α at the new positions;
    - the divergence solver, the diagnostic pressures, the divergence check, the walls.

    `max_iterations` = 0 runs neither solver, as in Rust. α and κ are solver columns. The second build's lists may need
    more room after the advection, so only an out-of-memory there leaves a partial step. ADR 0002 #21.
  - **Parity:** 35 new class S rows, 1,508 in all. They cover M5 part a's scenes and Rust's compressed-block test,
    iteration counts 0 to 20, and threshold edge cases. They also cover tiny masses against a tiny rest density, which
    reach α's 1e-10 floor, and the errors.
    - All 34 rows of the first set passed on the first replay. `validate.py` transcribes DFSPH: 49,942 rows,
      0 failures.
    - No row advects a position to ±∞. Its saturated key would overflow the second build's `cx + 1`, which panics in
      the oracle's build; the diverging rows use NaN instead.
  - **Tests:** the four DFSPH Rust tests, and `test_grid_cfl_dt` and `test_shallow_cfl_dt`, two grid and shallow-water
    tests that live in `sph.rs` and had been missed. That makes 74 of the 79; the last five are the multiphase step's.
    Part a ported 23 tests, not the 22 its record says, so the counts since then were one short (68 after M5 part b).

### Performance
- **DFSPH** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M5 part c; Rust's criterion
  suite has no DFSPH group, so against a twin): 1.46× on a sparse 64-particle block. On a dense 625-particle block it
  is 3.8× to 1% and 5.35× with every round run.

### Added (Cyrius)
- **sph** (M5 part b): `src/sph_pcisph.cyr`, the solver's PCISPH step.
  - **`s.step_pcisph(ps, n, config, viscosity, max_iterations, max_density_error)`,** in Rust's order: the densities;
    the non-pressure accelerations; δ from the prototype particle with the fullest neighbourhood; at least one
    correction iteration; the final prediction; then the walls. Its four buffers are the solver's, grown before any
    particle changes. ADR 0002 #20.
  - **Parity:** 37 new class S rows, 1,473 in all. They cover M5 part a's scenes and iteration counts (0, 1, 5, 20),
    with thresholds every iteration meets, none meets, NaN and negative. They also cover the Rust tests' shapes and
    both divergence paths.
    - All 36 rows of the first set passed on the first replay. `validate.py` transcribes PCISPH: 49,907 rows,
      0 failures.
    - The mutation tests found one gap, closed by a row that draws no random numbers: no zero-iteration row needed
      its one forced iteration.
  - **Tests:** the four PCISPH Rust tests (68 of `sph.rs`'s 79, as recounted at part c), and the solver layout
    test now pins all eight columns.

### Performance
- **PCISPH** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M5 part b): 1.34–1.75× Rust's
  criterion `pcisph_step` on its sparse blocks. On a dense 625-particle block it is 4.7× at 1% compression and 6.3×
  through five iterations, since each iteration repeats the prediction and the predicted-density sums.

### Added (Cyrius)
- **sph** (M5 part a): `src/sph.cyr`, the SPH solver.
  - **`PvSphSolver`.** All its buffers are its own and reused: part b's neighbour grid, embedded, plus the densities,
    the snapshot, the accelerations (Rust allocates these per step) and Verlet's previous accelerations. It is built
    with `PvSphSolver_new`, `_with_surface_tension` or `_default`, and released with `_drop`.
  - **`s.step(ps, n, config, viscosity)`:** the WCSPH step on part b's f32 neighbour lists, with CSF surface tension
    and symplectic Euler or velocity Verlet (`use_verlet`, `reset_integrator`). It grows every buffer before changing
    a particle, so `PV_ERR_OUT_OF_MEMORY` leaves the particles as they were. `s.neighbor_lists()` returns a
    `PvNeighborLists` for M4 part d's functions. ADR 0002 #19.
  - **`PvSphSolver_adaptive_dt`,** the CFL timestep.
  - **Parity:** 66 new `sph.tsv` rows, 1,436 in all.
    - 49 class S step rows: a hydrostatic tank, a dam break by both integrators, a droplet with surface tension, a
      3D box, a splash, a block at x ≈ 10⁶ (the f32 keys, D3) and the errors. They include trajectories to 300 steps.
    - 17 adaptive-dt rows.
    - Every row passed on the first replay. `validate.py` transcribes the step on part b's grid transcription,
      caching each scene's trajectory: 49,870 rows, 0 failures.
  - **Tests:** 19 Rust tests, among them the five part d tests that build their lists with the solver, so 63 of
    `sph.rs`'s 79. Also the five solver integration tests (29 of 33), and two Cyrius-only groups: the solver's layout,
    and `reset_integrator`.

### Performance
- **The solver step** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M5 part a).
  - Against Rust's criterion `solver_step` and `solver_step_st`, whose blocks are sparse at h 0.05: 1.24–2.3×.
  - Against a twin on a dense 625-particle block: 3.2× by Euler or Verlet, 3.6× with surface tension. A third of
    Cyrius's step is the neighbour build.
  - Writing the force loop's three small helpers out was measured at about 6% and not kept.

### Added (Cyrius)
- **sph** (M4 part d, which completes M4): the physics extensions, in `src/sph_fns.cyr`.
  - **On caller CSR lists:** `update_viscoelastic` (Oldroyd-B), `update_heat`, `apply_delta_sph`,
    `apply_implicit_viscosity` and `compute_gradient_corrections`, with `apply_gradient_correction`.
    - The lists are (pointer, count) words, validated as Rust's `neighbor_lists_valid` does (ADR 0002 #18).
    - The functions return 1, or 0 with nothing changed where Rust warns or would abort.
  - **Per particle:** `update_phase_change`, `update_combustion`, `update_reaction` and `apply_contact_angle`.
    - Rust's `ReactionProvider` trait is a Cyrius trait, `PvReactionProvider`, taken as a `dyn`.
    - `impl PvReactionProvider for PvCombustionConfig` mirrors Rust's impl. Its `react` returns a `PvReaction`,
      since a call through a `dyn` returns one value.
    - The contact angle's cosine is correctly rounded.
  - **The secondary particles:** `PvSecondaryType`, `PvSecondaryParticle`, `generate_secondary_particles`, and
    `update_secondary_particles`, which compacts the survivors in place.
  - **The configs:** `PvViscoelasticConfig`, `PvHeatConfig`, `PvPhaseChangeConfig` (with `_water_ice`) and
    `PvCombustionConfig` (with `_default`).
  - **The 3×3 inverse, ledger L7.** `_pv_invert_3x3` transcribes the oracle's own `matrix_inverse` (hisab 1.4.0: LU
    with an absolute 1e-12 pivot test), allocation-free.
    - ganita's `mat_inv`, the planned conversion target, differs on 12 of the 60 inverse rows: it refuses non-finite
      entries and scales its threshold. The correction is on the ganita request.
  - **Parity:** 318 new `sph.tsv` rows, 1,370 in all.
    - 315 of the first 316 passed on the first replay. The other was a contact-angle row at a cosine glibc misrounds,
      which matches the corrected oracle (`sph.cr.tsv`, now 22 rows).
    - `validate.py` transcribes part d: 49,804 rows, 0 failures.
    - The mutation tests found two gaps, exact ties no random row hits, closed by two rows that draw no random
      numbers.
  - **Tests:** 11 Rust tests, `test_degenerate_inputs_no_panic` among them. The five that build their lists with
    `SphSolver::step` wait for M5.
  - **Surface parity** checks Rust traits now: a `pub trait X` must be a `trait PvX`, and an `impl X for T` an
    `impl PvX for PvT`.

### Performance
- **The physics extensions** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M4 part d;
  against a twin, since Rust's criterion suite times none of them).
  - The list functions run 8.4–9.1× Rust, and the gradient corrections 5.7×.
  - The per-particle updates run 4.3–9×, and generating secondaries 15× (its 40 KB result is a block mapped per call).
  - The 3×3 inverse runs 1.5× (Rust allocates its `Vec<Vec>`).
  - Measured A/B, rows unchanged:
    - the gradient corrections got 2.1× faster (1.90–1.95 → 0.89–0.91 ms for 625), their moment matrix held in nine
      locals instead of stack words;
    - `update_secondary_particles` got 2.2× faster (20.9–21.8 → 9.4–9.7 µs), storing each field instead of building
      a `PvVec3`.
  - Two of Rust's criterion groups that part c's record missed, now recorded: `sph_viscosity_force` 8.2× and
    `create_particle_block` 25–35×. They agree with part c's twins.

### Added (Cyrius)
- **sph** (M4 part c): `src/sph_fns.cyr`.
  - **The brute-force sums:** `compute_density`, `pressure_force` and `viscosity_force`. The forces return a `PvVec3`;
    an index outside the slice reads NaN where Rust panics.
  - **The free step, as `sph_step`** (ADR 0002 #17: one namespace serves pravash and its consumers, and a bare `step`
    is the likeliest to collide). Its two per-call buffers are one freelist block, freed before it returns.
    `PV_ERR_OUT_OF_MEMORY` arrives before anything changes, and `PV_ERR_DIVERGED` leaves the state Rust leaves.
  - **`create_particle_block`, `split_particle` and `merge_particles`.** The block and the children are freelist
    buffers the caller frees. The split ring's cosines and sines are correctly rounded (ADR 0004).
  - **Parity:** 198 new `sph.tsv` rows, every one passing on the first replay of the code as it stands:
    - 60 brute-force rows and 43 step rows (trajectories to 300 steps, a 3D box, the walls, every config error, three
      divergences);
    - 36 block rows and 30 merge rows;
    - 29 split rows, 5 of them against the corrected oracle `sph.cr.tsv`, at child counts whose ring angles glibc
      misrounds (an MPFR search).

    `validate.py` transcribes part c: 49,486 rows, 0 failures.
  - **Tests:** the 10 part c Rust tests, among them `test_sort_by_zorder` and `_preserves_data` (deferred from part
    b), and three Cyrius-only groups for the no-panic paths. Also the four SPH tests of `rust-old/tests/integration.rs`.

### Fixed
- **`merge_particles`' position and velocity** (found by the golden rows before release). Storing a struct-returning
  call straight into a field, with struct-field arguments, stored the argument: cyrius 6.7.18's open
  `2026-11-09-field-store-of-struct-call-copies-a-field-argument.md`. All 30 merge rows were wrong. Each call is now
  bound to a local, and pravash's numbers are on the issue. Surface parity's `field_store_scan` flags the pattern; it
  simply hadn't been run before the first replay.

### Performance
- **The part c functions** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M4 part c; against
  Rust's criterion `sph_density`, `sph_step` and `sph_pressure_force`, and a twin for the rest).
  - The brute-force sums run 4.9–9.5× Rust and the free step 5.2–8.1×: the codegen's per-operation cost, since in
    these blocks almost no pair is within h.
  - `split_particle` into 8 takes 138 µs: its ring angles at fl(k·π/2) are the correctly rounded sin and cos's slow
    path (ledger L1). Into 7 it takes 3.5 µs.
  - Measured A/B, rows unchanged:
    - `create_particle_block` got 1.6× faster (140–154 → 85–90 µs for 1,000): one template particle copied whole,
      not a 144-byte struct return per particle.
    - `merge_particles` got 1.3× faster (126–129 → 95–100 ns), written out.

### Added (Cyrius)
- **sph** (M4 part b): `src/neighbor.cyr`.
  - **The neighbour grid,** `PvNeighborGrid` (ledger L4): hisab 1.4.0's `SpatialHash` as `SphSolver` uses it,
    with its f32 cell keys reproduced bit for bit (D3), on sorted columns instead of a `HashMap`. `_pv_nb_build`
    makes the solver's neighbour lists exactly as its private `build_neighbors` does: the grid kept within
    `f32::EPSILON` of h, the 27 probes per particle, and the probes that drift into other cells at large
    coordinates. Nothing is allocated once the columns have grown.
  - **`sort_by_zorder`,** stable on (Morton code, input index) (ADR 0002 #3), with `_pv_msort`, the stable merge
    sort both users share (ledger L9).
  - **ADR 0002 row 16:** the grid's form, the i32 probe offset (wraps, as Rust's release builds do), and
    `sort_by_zorder` returning 1 sorted or 0 unchanged.
  - **Parity:** 50 new `sph.tsv` rows, all bit-exact: 30 neighbour-list rows and 20 z-order rows. The 2,460
    `oracle::sph_cell_key` rows of `math.tsv`, pending since M0, now pass.
    - `validate.py` transcribes the grid and the Morton codes: 49,288 rows, 0 failures.
    - Two corpus gaps the mutation tests found are closed by four rows that draw no random numbers, so no other row
      moved. One gap was a grid kept within `f32::EPSILON` of h with points on cell edges and at 4.5e6. The other
      was a z-order input whose minima are both positive.
  - **Tests:** `test_zorder_negative_coordinates`, and three Cyrius-only groups in `tests/sph.tcyr`: the merge sort's
    stability, the lists against a brute-force search, and the grid's layout. `test_sort_by_zorder` and
    `_preserves_data` wait for part c's `create_particle_block`.

### Performance
- **The neighbour build and the z-order sort** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md),
  M4 part b; against a Rust twin on hisab 1.4.0's `SpatialHash`).
  - Builds of 1,000 particles: 1.24 ms in 2D and 1.60 ms in 3D. Rust takes 0.81–0.84 ms, so that is 1.5× and
    1.9×.
  - `sort_by_zorder` of 1,000: 191 µs, 2.0× Rust's 95 µs.
  - Against the first version, pinned and interleaved, with the rows unchanged:
    - the builds got 2.6× and 2.2× faster. Each axis's probe keys are made once per particle, each cell's run is
      appended in bulk, and the cells are ordered z, y, x with their keys in a column, so one binary search plus a
      short scan finds a probe row's three cells.
    - the sort got 3.8× faster. The order is applied in place, cycle by cycle, with each record moved by
      assignment instead of `memcpy` (which copies a byte at a time), and the scratch is one block.

### Added (Cyrius)
- **sph** (M4 part a): `src/sph_kernels.cyr`.
  - **The kernels:** `kernel_poly6`, `kernel_spiky_grad`, `kernel_viscosity_laplacian`, `kernel_wendland_c2`,
    `_c2_grad`, `_c4` and `_c4_grad`, with `PvKernelCoeffs` (the private `KernelCoeffs`).
  - **The equations of state:** `equation_of_state` and `equation_of_state_tait` (the γ → 0 `ln` limit included).
  - **The non-Newtonian viscosity models,** the first data-carrying enum (ADR 0002 #15): `PvNonNewtonianViscosity`,
    `PvNonNewtonianModel` and its constructors, and `non_newtonian_viscosity_effective_viscosity`.
  - **`batch_poly6`, `batch_distance_squared`, `total_kinetic_energy` and `max_speed`.**
  - **Private constants, generated:** `scripts/oracle/src/t_private.rs` writes `src/constants_private.cyr` from each
    private `const`'s Rust literal (`_PV_SPH_MIN_DENSITY`, `_PV_GRID_MIN_SIZE`, …). It fails on any unlisted one,
    and `probe_tables.py` compiles each against its literal. The grid's hand-typed 4s became `_PV_GRID_MIN_SIZE`.
  - **Parity:** a new golden module, `sph.tsv`, 804 rows, bit-exact on the first replay.
    - It includes 471 Tait rows (12 at arguments where glibc misrounds pow or ln) and 22 viscosity-model rows.
    - `validate.py` transcribes part a and writes the corrected oracle `sph.cr.tsv` (16 rows). The port matches
      every one.
  - **Tests:** 22 Rust tests in `tests/sph.tcyr`. Surface parity adds `sph`, with the rest of it listed as pending
    by part.

### Performance
- **The SPH kernels** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M4 part a; against
  Rust's criterion `sph_kernels` group and a twin for the rest).
  - poly6 21 ns (Rust 2.03 ns), the classic kernels 10–10.5×, Wendland 8–9×.
  - Tait 52× and the power law 27×: the correctly rounded `pow` (L1).
  - Each public kernel computing only its own coefficient made the classic kernels 1.30–1.78× faster with the rows
    unchanged.

### Added (Cyrius)
- **The worker pool and the grid's thread count** (M3 part e, which completes M3; port plan D6).
  - **`src/par.cyr`:** `PvPool_new`, `PvPool_drop` and `_pv_par_for`.
    - The caller plus T − 1 persistent workers on `thread_create` and `chan_*`, each running one contiguous slice of
      an index range.
    - Serial where threads can't run (agnos, cx) or can't be started.
    - The pool loads crmath's tables before starting workers.
  - **`PvFluidGrid.set_threads(n)`** (ADR 0002 #4, amended). The count is 1 by default, and the first step after a
    change starts the pool. A clone keeps the count, and drop stops the pool. `PvFluidGrid` is now 376 bytes.
  - **The three loops Rust runs on rayon** go through the pool: the advection, the periodic advection and the
    projection, each row written by one thread.
  - **Parity:** `tests/parity.tcyr` replays every `FluidGrid::step` row at 2, 3, 4 and 8 threads and fails one whose
    words differ from its one-thread run. Result: 376 rows, 0 differing, on x86_64 ELF, PE and aarch64 and under
    `--bounds`.
  - **Tests:** `tests/par.tcyr`, 40 checks: exact coverage on T distinct threads, short and empty ranges, no heap
    per call, the grid's thread-count lifecycle.

### Performance
- **Thread counts** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M3 part e; cores 8–15).
  - The pooled periodic advection runs 2.0× faster at 2 threads, 3.2–3.6× at 4 and 4.1× at 8 (255 × 255).
  - A whole step gains 1.07× (63 × 63) to 1.15× (255 × 255) at 8 threads: its serial Gauss–Seidel solves dominate,
    in Rust too.
  - Rust's own rayon build is no faster than its serial one at 63 × 63 (`grid_step_periodic` 2.07 ms either way)
    and slower on small grids (124 µs against 70.6 µs at 15 × 15).

### Added (Cyrius)
- **vof** and **phase_field** (M3 part d): `src/vof.cyr` and `src/phase_field.cyr`, the ports of `rust-old/src/vof.rs`
  and `phase_field.rs`.
  - **`PvVofField`:**
    - `PvVofField_new`, `_clone` and `_drop`;
    - `is_surface`, `is_full`, `is_empty` and their `_with_threshold` forms;
    - `fill_rect`, `fill_circle`, `total_volume` and `interface_normal`;
    - `advect` (donor-cell fluxes, closed walls, sub-cycled to Courant 0.5);
    - JSON.
  - **`PvPhaseField`:**
    - `PvPhaseField_new`, `_clone` and `_drop`;
    - `set_circle`, `set_rect`, `is_phase_b`, `is_interface` and `phase_b_area`;
    - the Allen–Cahn `step` (upwind advection, zero-gradient ghosts);
    - JSON.
  - **ADR 0002 #14:**
    - (pointer, count) velocities;
    - every size product checked where Rust panics;
    - the scratch grown before any change;
    - NaN for a cell outside a field.
  - **Parity:** two new golden modules, `vof.tsv` (224 rows) and `phase_field.tsv` (160), bit-exact on the first
    replay.
    - They include 63 `VofField::advect` and 75 `PhaseField::step` rows (class S, bound 0), with 300-step
      trajectories, 1 to 1,024 sub-steps, unchecked cell sizes and every refusal.
    - `validate.py` transcribes both modules and recomputes every row.
  - **Tests:** all 8 Rust tests of `vof.rs` (`tests/vof.tcyr`) and all 11 of `phase_field.rs`
    (`tests/phase_field.tcyr`), with layout groups.
  - **Surface parity:** both modules added, 258 items ok.
  - **Mutation checks:**
    - 11 of 14 port mutations fail rows, 4 of them through rows added when they first survived.
    - Two of the rest are equivalent, and one changes only a signed zero, which classes B, C and S treat as equal.
    - All 3 transcription mutations are caught.

### Performance
- **The part d benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M3 part d),
  against Rust twins on the same bodies (`scripts/probes/rust/src/bin/bench_vof_pf.rs`). Pinned, interleaved:
  - `vof_advect` 54.5 µs (Rust 6.77 µs) at 32 × 32, 8.1–8.7×;
  - `phase_field_step` 41.3 µs (4.65 µs), 8.7–8.9×.

  Writing out the VOF sub-step's flux and clamp calls, and computing each shared x face once, made the advection
  1.17–1.18× faster. Writing out the phase-field clamp made its step 1.18–1.19× faster. Both keep the rows unchanged.

### Fixed
- **`scripts/oracle/src/m_grid_c.rs`** was not rustfmt-clean (one tuple line, after a late edit in part c).

### Added (Cyrius)
- **grid** (M3 part c): the rest of `rust-old/src/grid.rs`.
  - **`src/k_epsilon.cyr`:** `PvKEpsilon`, the k-ε model:
    - `PvKEpsilon_new`, `_clone` and `_drop`;
    - `turbulent_viscosity`;
    - the source-term `step`, which takes (pointer, count) slices and returns 1, or 0 where Rust warns;
    - JSON.
  - **`apply_ghost_fluid`** (`src/grid.cyr`): the level-set pressure correction, with the same flag.
  - **`src/mac_grid.cyr`:** `PvMacGrid`, the staggered grid:
    - `PvMacGrid_new`, `_clone` and `_drop`;
    - `velocity_at`, `max_speed`, `divergence_at` and `cfl_dt`;
    - the step (closed walls, face buoyancy, Neumann Gauss–Seidel with the mean pinned, density advection);
    - JSON.
  - **`_pv_col_swap`** (`src/column.cyr`), Rust's `std::mem::swap` of two Vecs.
  - **ADR 0002 #13:**
    - the flags;
    - a `KEpsilon::new` that can fail;
    - every size product checked where Rust panics on overflow;
    - NaN for an out-of-range read.
  - **Parity:** `grid.tsv` grows to 1,120 rows. Part c's 293 rows, from their own seed, are bit-exact on the first
    replay:
    - 46 `KEpsilon::step` and 89 `MacGrid::step` rows (class S, bound 0), trajectories up to 300 steps among them;
    - the constructors, the reads and the ghost fluid;
    - JSON for both types.
    - `validate.py` transcribes the three and recomputes every row.
  - **Tests:** the last five Rust tests of `grid.rs` and the MacGrid half of `test_cfl_dt_finite_without_flow`.
    `tests/grid.tcyr` now has all 49, with 218 checks, and the layout group pins both new records.
  - **Surface parity:** all of `grid`, 227 items ok, 0 pending.
  - **Mutation checks:** 8 of 12 port mutations fail rows.
    - Three of the rest are provably equivalent.
    - The ghost Laplacian's re-association shows as 1–2 ulp in the histogram, inside class B's bound
      (port-parity §12).
    - All 3 transcription mutations are caught.

### Performance
- **The part c benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M3 part c),
  against Rust twins on the same bodies (`scripts/probes/rust/src/bin/bench_grid_c.rs`; Rust has no criterion bench
  for these). Pinned, interleaved:
  - `mac_step` 210 µs (Rust 73.6 µs) at 16 × 16, 2.2–2.9×;
  - `ghost_fluid` 14.6 µs (1.97 µs) at 32 × 32, 7.0–7.4×;
  - `kepsilon_step` 61.1 µs (4.30 µs) at 32 × 32, 13–14×. Its three maxes are written out, which made the step 1.21×
    faster (74.0 → 61.1 µs) with the rows unchanged. The rest is the loop Rust vectorises.

### Added (Cyrius)
- **grid** (M3 part b): the DST and CG projections under parity, and the DST plans.
  - **Parity:** `tests/golden/grid.tsv` grows to 827 rows. Part b's rows come from their own seed, so part a's 569
    keep their values. Every row is bit-exact against the oracle, or against the corrected oracle where glibc
    misrounds one of the DST's sines or cosines:
    - `oracle::pressure_solve_dst`: 21 rows, class C, bound 0. They cover part a's shared systems, part b's systems
      at the transform lengths where glibc misrounds (12, 18, 25, 30, 36, 38, 60), and the edges.
    - `oracle::pressure_solve_cg`: 49 rows, class B, at 1 to 200 iterations, including the empty system hisab
      refuses.
    - 188 more `FluidGrid::step` rows on the DST and CG projections: 300-step trajectories of still tanks; single
      steps of plumes (10 × 10, and 14 × 14 where the DST misrounds), a vortex and a jet under every scheme and
      force; CG at budgets 1 to 200; random states.
    - `tests/golden/grid.cr.tsv` (ADR 0004): 20 rows (6 DST solves, 14 steps), written by `validate.py` from MPFR.
      `validate.py` now transcribes hisab 1.4.0's `dst`, `idst` and `conjugate_gradient` and recomputes every
      grid row.
  - **DST plans:** `PvFluidGrid` gains two Cyrius-only private columns, `dst_x` and `dst_y` (ADR 0002 #12,
    amended), and is now 360 bytes.
    - Each holds a transform length's n² sines and n eigenvalue terms. The first DST step builds them, and later
      steps keep them while the interior keeps its size.
    - Rust computes the same values inside every sum, so the bits are the same.
    - `src/grid_solve.cyr`'s `_pv_dst` / `_pv_idst` take the sine table, built by `_pv_dst_sines`.
  - **Tests:** `fg_dst_plan_is_kept_per_length` in `tests/grid.tcyr`, which now has 185 checks.
  - **Mutation checks:** 8 of 9 port mutations fail rows or the plan test. The ninth is provably equivalent: a
    factor of 2 commutes with rounding. All 3 mutations of the transcription are caught.

### Performance
- **The DST step** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M3 part b; pinned,
  interleaved A/B; the bits unchanged). Rust 1.3.0's numbers are in parentheses.
  - `grid_step`:
    - 28.5 ms → 760 µs at 15 × 15 (127 µs);
    - 15.9 → 1.10 ms at 30 × 30 (1.07 ms);
    - 719 → 8.6 ms at 63 × 63 (10.1 ms).
  - A grid's later steps reuse its plans (`grid_step_warm`, Cyrius-only): 190 µs, 965 µs and 5.64 ms, which is
    0.56–1.5× Rust.
  - The first 15 × 15 step's plan costs about 570 µs. Most of that is 25 sines on the correctly rounded slow path
    (ledger L1).
  - Appended to hisab's L6 request: a DST into caller scratch that takes a plan the caller keeps.

### Fixed
- **`cyrius audit`** (a CI step) failed on undocumented public functions: 43 test and bench runners and helpers
  (`main`, the `test_*` groups, `near`, the `bench_*` sections). Each now has its doc comment, and the audit's fmt,
  lint, docs, tests and bench are clean.

### Added (Cyrius)
- **grid** (M3 part a): `src/grid.cyr` and `src/grid_solve.cyr`, the port of `rust-old/src/grid.rs`'s `FluidGrid`.
  - **`PvFluidGrid`:**
    - `PvFluidGrid_new`, `PvFluidGrid_clone` and `PvFluidGrid_drop`;
    - the reads `cell_count`, `idx`, `velocity_at`, `speed_at`, `max_speed`, `cfl_dt` and `total_kinetic_energy`;
    - the public `PvFluidGrid_diffuse` (Rust's `FluidGrid::diffuse`);
    - JSON (`to_json`, `PvFluidGrid_from_json`).
  - **`PvGridConfig`** (`PvGridConfig_smoke`, `PvGridConfig_default`) and **`PvBoundaryCondition`** (NoSlip 0,
    FreeSlip 1, Periodic 2), each with its JSON.
  - **`step`:**
    - semi-Lagrangian, periodic, MacCormack and BFECC advection, with the backtrace limiter;
    - Gauss–Seidel diffusion and the Smagorinsky viscosity;
    - buoyancy and vorticity confinement;
    - the CG, multigrid, Gauss–Seidel and DST projections;
    - the ghost rules for each wall type.

    Every formula follows Rust's operation order.
  - **`src/grid_solve.cyr`:** hisab 1.4.0's direct DST-I / IDST-I (ledger L6) and matrix-free linear CG (L5),
    transcribed into caller scratch.
  - **Storage:** the public fields are `PvColumn`s, the multigrid levels persist as `PvMgLevel` records, and three
    Cyrius-only private buffers replace the DST's and CG's per-call allocations (ADR 0002 #12). A step allocates
    nothing once sized.
  - **`PvGridConfig.boundary` outside the enum** is refused with `PV_ERR_INVALID_PARAMETER` (ADR 0002 #12).
  - **Parity:** `tests/golden/grid.tsv`, 569 rows, all bit-exact on the first replay on three targets, and the same
    with rayon on.
    - 188 of them are `FluidGrid::step` rows (class S, bound 0) on the periodic and multigrid projections: 300-step
      trajectories of still tanks, and single steps at frames {0, 1, 2, 10, 60, 300} of plumes, a vortex and a jet
      under every scheme and force.
    - 70 rows run the Gauss–Seidel and multigrid oracle hooks on shared systems. The DST and CG rows come with part b.
    - `validate.py` transcribes the grid and recomputes every row.
  - **Tests:** 44 of the 49 Rust tests of `grid.rs` (`tests/grid.tcyr`; the other 5 need part c's MacGrid and
    KEpsilon), the 5 grid integration tests, and `serde_roundtrip_all_types`.
  - **Surface parity:** 227 items, 24 pending part c.
- **`_pv_fmod`** (`src/mathx.cyr`): the exact fmod, Rust's `%` on f64, transcribed from hisab 3.4.1. It is ledger
  L14, a stand-in until ganita has one (requested, `ganita:docs/development/issues/2026-11-11-no-fmod.md`).
  `tests/golden/math.tsv` gains 810 `f64::rem` rows, all bit-exact.

### Performance
- **The grid benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M3 part a),
  against Rust's own criterion benches, re-run pinned:
  - `grid_step_periodic` 284 µs (Rust 70.6 µs) at 15 × 15, 2.7–4.0×;
  - `grid_diffuse` 320 µs (86.5 µs) at 32 × 32, 2.5–3.7×;
  - `grid_max_speed` 5.89 µs (360 ns), 16–17×;
  - `grid_step` (the DST projection) 29.3 ms (127 µs) at 15 × 15 and 736 ms (10.1 ms) at 63 × 63.

  `max_speed` with its max written out went from 8.56 µs to 5.95 µs (pinned A/B), its rows unchanged. The DST steps'
  cost is the correctly rounded sin at fl(k·π/2), 18.4 µs a call there against 165 ns elsewhere. That is appended
  to the L1 request, and part b's per-length sine table keeps the bits and removes it.

### Added (Cyrius)
- **shallow** (M2): `src/shallow.cyr`, the port of `rust-old/src/shallow.rs` and the first stateful solver.
  - **`PvShallowWater`:**
    - `PvShallowWater_new`, `PvShallowWater_clone` and `PvShallowWater_drop`;
    - the cell reads `depth_at`, `surface_at`, `is_wet` and `is_breaking`;
    - `add_disturbance`, `total_volume`, `interior_volume`, `max_wave_height` and `cfl_dt`;
    - JSON (`to_json`, `PvShallowWater_from_json`).
  - **`step`:**
    - the default well-balanced forward-backward scheme;
    - the HLL finite-volume path (`use_riemann`);
    - Peregrine's implicit dispersion and its Green–Naghdi variant;
    - Manning friction, breaking, the damping `powf` and the closed walls.

    Every formula follows Rust's operation order, on the correctly rounded powf and cbrt.
  - **Sediment:** `PvSedimentConfig` and `update_sediment`.
  - **Storage:** the public fields are `PvColumn`s (`PvColumn` gains `resize`, `fill`, `copy_from`, `at`). The
    scratch buffers persist, so a step allocates nothing once sized.
  - **Errors:** an allocation failure is the Cyrius-only `PV_ERR_OUT_OF_MEMORY` (11; ADR 0002 #11).
  - **Parity:** `tests/golden/shallow.tsv`, 888 rows, all bit-exact on the first replay on three targets.
    - 142 of them are `ShallowWater::step` rows in the new **class S** (single solver steps; `_par_c_bound` gives the
      bound, 0 here).
    - The non-chaotic scenes are 300-step trajectories, and the rest single steps from the oracle's own state at
      frames {0, 1, 2, 10, 60, 300}.
    - `validate.py` transcribes the whole step and reproduces every row on glibc. MPFR finds no correction: glibc's
      pow is correctly rounded on all 22 damping pairs used.
  - **Tests:** all 44 Rust tests of `shallow.rs` (`tests/shallow.tcyr`) and the 3 shallow integration tests.
  - **Surface parity:** 165 items.

### Performance
- **The shallow-water benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M2),
  against Rust's own criterion benches, re-run pinned:
  - `shallow_step` 221 µs (Rust 30.7 µs) at 32 × 32, 7.2× at every size;
  - `shallow_volume` 4.75 µs (773 ns);
  - `shallow_disturbance` 8.45 µs (5.17 µs).

  `total_volume` on raw loads with the max written inline went from 14.6 µs to 4.75 µs (pinned A/B), its rows
  unchanged. The step's remaining cost is calls to small helpers that `#inline` declines (filed at cyrius).

- **buoyancy, vortex, bridge, json** (M1 part 4, which completes M1).
  - **`src/buoyancy.cyr`:**
    - `buoyancy_force` and `drag_force`;
    - `terminal_velocity` and `reynolds_number`, which return an f64 `Result`;
    - `PvFlowRegime` with `classify_flow`.
  - **`src/vortex.cyr`:** `vorticity_2d`, `lamb_oseen_velocity`, `rankine_velocity`, `enstrophy` (Rust's `Sum`:
    an empty field is −0) and `kolmogorov_scale`.
  - **`src/bridge.cyr`:** the ten pavan / ushma / goonj conversions.
  - **Exact arithmetic:** each function follows Rust's operation order, and the four that call a transcendental use
    the correctly rounded primitives (ADR 0004).
  - **`src/json.cyr`:** serde-compatible JSON for `FluidMaterial`, `FluidParticle`, `FluidConfig`, `ParticleSoa`,
    `FlowRegime` and `DragCoefficient`.
    - `x.to_json(sb)` writes, and `PvX_from_json(buf, len, out)` reads with serde's derive semantics (both struct
      forms, unknown fields ignored, the u8 range, `null` refused for an f64).
    - Every serde_json document decodes to the same bits, and every Cyrius document round-trips. The digits are
      bayan's Grisu2, not serde_json's text (D4).
    - It is a marked stand-in (`UPSTREAM(L10)`). The new Cyrius-only code `PV_ERR_DESERIALIZE` (10) comes with ADR
      0002 #10.
  - **Parity:** 8,275 new golden rows in `buoyancy.tsv` (1,716), `vortex.tsv` (2,293), `bridge.tsv` (3,964) and
    `json.tsv` (302).
    - Every row is recomputed by `validate.py`. The class C functions use glibc for the record and MPFR for the
      corrected oracle, which gains `vortex.cr.tsv` (3) and `bridge.cr.tsv` (26).
    - All rows pass on the first replay, on three targets.
    - Three JSON documents that bayan's policies read otherwise than serde_json (S3, S4, the integer `-0`) are rows
      of their own, pending until bayan changes.
  - **Tests:**
    - all 45 Rust tests of the three modules;
    - `common.rs`'s two serde tests, so all 49 of its tests are ported;
    - 11 of `rust-old/tests/integration.rs`, in the new `tests/integration.tcyr`.
  - **Surface parity:** 128 items.

### Performance
- **The M1 part 4 benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md)), against
  Rust 1.3.0:
  - `terminal_velocity` 50 ns (Rust 3.6), `lamb_oseen_velocity` 136 ns (7.4), `enstrophy` of 1,000 values 3.0 µs
    (759 ns);
  - JSON about 20× serde_json: a config read 7.8 µs (329 ns), a 100-particle SoA read 572 µs (31.9 µs).

  The class C rows cost their correctly rounded primitive (ledger L1).

- **error, vec3, common** (M1 part 3): the shared fluid types, the port of `rust-old/src/error.rs` and
  `common.rs`.
  - **`src/error.cyr`:** the `PV_ERR_*` codes in Rust's variant order, append-only, with `pv_error_name` and
    `pv_error_message` (ADR 0002 #1).
  - **`src/vec3.cyr`:** `PvVec3` with glam 0.29.3's `dot`, `length`, `distance` and `is_finite`, in glam's order. It
    is a marked stand-in (`UPSTREAM(L3)`).
  - **`src/common.cyr`:**
    - `PvFluidMaterial`, with the five presets and `custom`;
    - `PvFluidParticle`, a 144-byte record laid out as Rust's;
    - `PvFluidConfig`, with `water_2d`, `validate` and `check_cfl`;
    - `PvParticleSoa`, 18 columns;
    - `PvArenaHandle` and `PvParticleArena`, whose free list is sorted and merged, whose handles go stale after
      `compact`, and which allocates nothing after `new`.
  - **`src/column.cyr`:** the freelist column the SoA's `Vec` fields become. It frees on growth and never aborts. It
    is a marked stand-in (`UPSTREAM(L13)`), because the stdlib `vec` keeps outgrown buffers and aborts on a bad
    index (filed).
  - **API shapes:** ADR 0002 rows 8 and 9.
    - Nothing aborts: a Rust panic becomes NaN, 0 or an empty view.
    - `Option<ArenaHandle>` is an out pointer plus 1 / 0, and a slice is (pointer, count).
    - Destructors are `_drop`.
  - **Parity:** the golden module `tests/golden/common.tsv`, 3,683 rows, 59 of them arena scenarios.
    - `validate.py` recomputes every row, with a Python transcription of the arena.
    - All rows are bit-exact on the first replay, on x86_64 ELF, x86_64 PE and aarch64.
    - The replay is 34,454 ok and 43 accepted (ADR 4).
  - **Tests:** `tests/common.tcyr` ports 47 of `common.rs`'s 49 Rust tests; the two serde tests come with part 4's
    JSON. `tests/error.tcyr` ports the six error tests.
  - **Surface parity:** `scripts/surface_parity.py` (impetus's) checks every public Rust item of the ported modules
    against its Cyrius name (98 items), and now runs in CI.

### Performance
- **The `common` benchmarks** ([`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md), M1 part 3),
  against a Rust 1.3.0 twin (`scripts/probes/rust/src/bin/bench_common.rs`).

  | Bench | Rust 1.3.0 | Cyrius |
  |---|---|---|
  | `soa/from_aos_1000` | 16.9 µs | 206 µs |
  | `soa/write_to_aos_1000` | 6.5 µs | 36 µs |
  | `arena/cycle_1000` | 8.3 µs | 110 µs |
  | `common/config_validate` | 12.5 ns | 64–100 ns |

  Two changes, measured A/B (pinned, interleaved):
  - **Record resets** copy one default record: `arena/cycle_1000` from 163–183 µs to 109–126 µs.
  - **The SoA conversions** move words straight between records and columns: `from_aos` 1.8×, `to_aos` 1.9×,
    `write_to_aos` 3.6× faster.

  The rest is the freelist mapping each block over 4 KB, which is filed upstream.
- **Ledger L12 closed, no stand-in needed.** A struct assignment copies a 144-byte particle in 11.7 ns, against
  `memcpy`'s 245 ns. Every particle move is an assignment. The numbers are on the open cyrius `memcpy` issue.

- **crmath and crlog** (M1 part 2, [ADR 0004](docs/adr/0004-correct-rounding-is-the-accuracy-target.md)): every
  std transcendental pravash calls is now correctly rounded, the same bits on every target. Each file is a marked
  stand-in (`UPSTREAM(L1)`) until ganita's correctly rounded tier ships.
  - **The code:**
    - `src/crmath.cyr`: impetus's exp, sin, cos, sin_cos, hypot, pow and atan2;
    - `src/crlog.cyr`: kimiya's ln and log10;
    - their tables `src/crmath_data.cyr` and `src/crlog_data.cyr`, generated by `scripts/oracle/crtables.py` and
      `lntables.py` and checked against MPFR;
    - `src/mathx.cyr` wraps them as `_pv_exp`, `_pv_ln`, `_pv_log10`, `_pv_powf`, `_pv_sin`, `_pv_cos`,
      `_pv_sin_cos`, `_pv_hypot` and `_pv_atan2`, plus `_pv_cbrt` (ganita's, already exact).
  - **The corrected oracle:** `tests/golden/math.cr.tsv`, the 43 math rows glibc 2.44 misrounds (21 log10,
    11 hypot, 3 sin, 3 sin_cos, 2 atan2, 1 each of exp, ln and powf), with MPFR's values.
    - `scripts/oracle/validate.py --write` writes it and the two table files; a plain run checks all three.
    - MPFR comes through `scripts/oracle/mpfr_ref.py`.
  - **Parity:** K = 0 for every class C math function, powf and log10 included.
    - The replay is 30,771 ok and 43 accepted (ADR 4), with 0 failed and 0 measured, identical on x86_64 ELF,
      x86_64 PE and aarch64.
    - The harness core is now impetus's current version, which compares corrected rows.
    - Mutation-checked: a 1-ulp exp fails 164 rows; a one-bit table edit fails rows and `validate.py`.
  - **Tests:** `tests/crmath.tcyr`, 85 checks covering the specials, the exact cases and ties, the hard-to-round
    arguments, and the agreement between fast and slow paths, on all three targets.

### Performance
- **The first benchmark record**, [`docs/benchmarks-rust-v-cyrius.md`](docs/benchmarks-rust-v-cyrius.md):
  `tests/pravash.bcyr`'s `math/*` rows against a Rust twin (`scripts/probes/rust/src/bin/bench_libm.rs`), on the same
  1,024 seeded arguments per function, pinned.

  | Function | Correctly rounded (Cyrius) | Counterpart it replaced (Cyrius) | Rust (glibc) |
  |---|---|---|---|
  | ln | 58 ns | 62 ns | 4.64 ns |
  | log10 | 94 ns | — (none existed) | 8.54 ns |
  | exp | 110 ns | 55 ns | 4.85 ns |
  | sin_cos | 228 ns | 82 ns | 12.3 ns |
  | hypot | 173 ns | 33 ns | 6.23 ns |
  | powf | 757 ns | 212 ns | 12.2 ns |
  | atan2 | 1.41 µs | 63 ns | 13.6 ns |

  atan2's mean comes from its slow path: 302 µs a call, on 0.4 % of arguments. Phase 1 alone is 349 ns.

- **mathx** (M1 part 1): `src/mathx.cyr`, the library's first module after the generated constants. It holds
  pravash's Rust-semantics f64 helpers: `_pv_min` / `_pv_max` / `_pv_clamp` (NaN-ignoring, Rust's ±0 rule),
  `_pv_is_nan` / `_pv_is_finite`, `_pv_to_radians`, `_pv_powi2` / `_pv_powi8` in LLVM's order, and the saturating
  `_pv_as_i32` / `_pv_as_u32` / `_pv_as_usize`. It is a marked stand-in (`UPSTREAM(L2)`).
  - **Build:** the first `[lib]` in `cyrius.cyml` (constants, mathx), the bundle `dist/pravash.cyr`, and
    `tests/lib.cyr` for test compiles.
  - **Parity:** 6,371 golden rows newly gated, all bit-exact. The replay is 28,223 ok, 0 failed, identical on
    x86_64 ELF, x86_64 PE and aarch64. Mutation checks: the ±0 rule fails 2 rows, unsaturated negatives 111.

### Changed
- **The Cyrius port begins** (port plan M0 steps 1–3; ADRs 0001–0003). `cyrius port` moved the Rust 1.3.0 crate,
  with its `oracle` hooks and `Cargo.lock`, into `rust-old/` as the frozen parity oracle (commit `a88bc4e`); it
  still passes there. The repairs:
  - the Cyrius manifest: `cyrius.cyml` reads `VERSION`, `[build] dce`, the stdlib leaves, hisab 3.4.1 test-scoped;
  - CI: the Cyrius workflow runs build, test, `test --bounds`, `deps --locked` and `audit`, plus a job that keeps
    the frozen oracle green with and without `parallel`;
  - docs: CLAUDE.md merged for Cyrius; `state.md`; ADR 0001 (port with a Rust oracle), 0002 (API differences)
    and 0003 (local stand-ins for library gaps); CONTRIBUTING and SECURITY for Cyrius; a port banner in the README;
  - archived in `rust-old/`: the Rust workflows, scripts, architecture overview and 1.2 → 1.3 migration guide.

  The library itself is still the Rust 1.3.0 crate: no Cyrius module exists yet.

### Added
- **Semantics probe** (port plan M0 step 4): `scripts/probes/p0_semantics.cyr`, its Rust twin `p0_semantics.rs`
  and `p0_compare.py` compare the bits of the Cyrius counterpart of every f64 / f32 operation pravash uses
  (`docs/development/port-parity.md` §1).
  - **Bit-identical:** the f32 SPH cell keys (including saturation and NaN), floor / ceil / sqrt / abs, ln,
    sin / cos, hypot, cbrt, `to_radians` and `powi`.
  - **Need helpers:** `f64_min` / `f64_max` (asymmetric NaN), and `f64_to`, which doesn't saturate as Rust's
    `as i32` / `as u32` / `as usize` do.
  - **Correctly rounded stand-ins needed:** log10 (there is none), and ganita's pow and atan2 and the exp
    polyfill (1 ulp off).
- **Golden-value generator** (port plan M0 step 5): `scripts/oracle/`, adapted from impetus's. It drives the frozen
  oracle and writes `tests/golden/`: `constants.tsv` (29 rows) and `math.tsv` (33,239 rows).
  - **Format and determinism:** TSV format 1, every float as its bits. Two runs give the same bytes, and the
    `parallel` build reproduces the corpus but may only `--check` it.
  - **`--check`:** refuses a corpus recorded on another glibc.
  - **`validate.py`:** recomputes every row without the oracle. Class A/B in Python; glibc's class C through
    ctypes; `cbrt` proven correctly rounded.
  - **Finding:** the oracle's `cbrt` is compiler_builtins', not glibc's (port-parity §3).
- **`src/constants.cyr`** (port plan M0 step 6), generated by `scripts/oracle` (never edited by hand). It holds
  the 14 public constants (`PV_FLUID_MATERIAL_*`, `PV_DRAG_COEFFICIENT_*`, `PV_ADDED_MASS_COEFFICIENT_*`) and the
  6 Rust std float constants the code uses (`PV_F64_MAX`, `PV_F64_INFINITY`, `PV_F64_NEG_INFINITY`, `PV_F64_NAN`,
  `PV_F64_PI`, `PV_F32_EPSILON`).
  - **Proofs:** both lists are proven complete against `rust-old/src`, and `constants.tsv` pins all 35 values.
  - **`scripts/oracle/probe_tables.py`:** compiles each value and confirms its bits; a 2-ulp change to one literal
    fails it.
- **Parity harness** (port plan M0 step 7): `tests/parity.tcyr` replays `tests/golden/` against the port, with
  impetus's `tests/parity_core.cyr` and `tests/ulp.cyr`. The constants rows go through the generated
  `tests/constants_rows.cyr`.
  - **First run:** 0 failing; 5,149 ok, 0 failed, 18,493 measured (class C, ≤ 2 ulp, bounds at step 8), and
    9,632 pending (the M1 helpers). It is warning-free, and the same under `--bounds`.
  - **Finding:** ganita's cbrt equals the oracle's on all 1,025 rows, so cbrt needs no stand-in.
  - **Mutation-checked:** a flipped golden bit, a 1-ulp constant, and a mis-wired dispatcher each fail their row.
- **Cross-target baselines** (port plan M0 step 8): per-function digests in `tests/parity.tcyr`. Every dispatched
  function gives identical bits on x86_64 ELF, x86_64 PE (wine) and aarch64 (qemu); only the native x87 shadows
  differ, by design.
  - **Bounds:** K = 2 for sin, cos, sin_cos, atan2, exp, ln and hypot (measured max 1, hypot 2), and K = 0 for
    cbrt. powf stays measured until its correctly rounded M1 counterpart.
  - **Gated:** 21,852 ok, 0 failed on all three targets. Both bounds are mutation-checked.
- **Probes** (port plan M0 step 9; `scripts/probes/`, `docs/development/port-parity.md` §4):
  - **Memory:** every hisab or ganita return leaks on the step path. The neighbour-hash rebuild leaks 4.1 MB per
    1,000-particle step, `num_dst` 10–20 KB per call, and the 3×3 `mat_inv` 248 B. Freelist and a step arena
    are flat.
  - **Layout:** all four forms give the same bits. AoS with f64 locals is fastest (12.9 ns a pair); struct
    values are 2.5× slower and f64v4 3.5×.
  - **f32 keys:** 10⁶ of 10⁶ match Rust on x86_64 ELF, x86_64 PE and aarch64.
  - **Neighbour order:** (cell offset, insertion); at |x| ≈ 10⁶, 89% of the probe points miss their cell, the
    defect D3 fixes.
  - **Sort ties:** Rust's ipnsort leaves a different order at 75–94% of positions on tie-heavy inputs, so the
    stable (key, index) order applies (ADR 0002 #3).
  - **Parallel:** the Rust oracle's serial and parallel builds give the same digests on 14 scenarios. A Cyrius
    persistent pool gives the serial bits at 1–8 threads, allocates nothing, and is 3.9× at 8 for 10⁵ particles.
  - **Bundle:** DCE takes a full-hisab consumer from 3.06 MB to 677 KB, so there is one bundle (ADR 0002 #5).
  - **Libraries vs hisab 1.4.0:** hisab 3.4.1's DST is another algorithm, so ledger L6 transcribes 1.4.0's.
    ganita's `mat_inv` gives the oracle's bits on all 300 matrices, so L7 is needed for memory only. The
    autodiff tape is within 3 ulp, so it stays on hisab as class D.
- **Upstream requests and the conversion ledger** (port plan M0 step 10; ADR 0003). Each request was filed with
  pravash's numbers and, where the gap can be shown, a self-proving repro verified from the owning repo's root:
  - **hisab:** a non-allocating spatial-hash query (4.1 MB per 1,000-particle step today), a matrix-free linear CG,
    and an allocation-free DST;
  - **ganita:** `mat_inv_into`;
  - **cyrius:** NaN-ignoring min / max with the Rust f64 surface, a stable sort, `#derive` array and tagged
    shapes, and the struct-value / `f64v4` costs;
  - **sakshi:** a chrome://tracing export;
  - **existing issues:** pravash's numbers added to ganita's correctly rounded tier, hisab H9, cyrius S1 and the
    `cyrius port` clobber issue.

  The ledger (plan §8) is seeded with 11 open rows and 1 conditional. `scripts/upstream_ledger.py --check`, run in
  CI, ties every `UPSTREAM(Ln)` note in `src/` to an open row, every existing stand-in to its note, and every row
  to its filed request. **M0 is complete.**

## [1.3.0] - 2026-11-09

Toolchain/dependency refresh plus a full correctness audit of every module.
Most fixes change numerical results — read **Behavior changes** before
upgrading. No public signatures were removed or changed.

### Behavior changes (migration)
- **grid**: pressure projection now actually removes divergence for any `dx` (it amplified divergence ~1/dx² for dx < 1). Simulations at dx ≠ 1 behave very differently — re-tune forces tuned against the old behaviour.
- **grid**: `FreeSlip` ghost cells now reflect the normal velocity (`ghost = −interior`, zero at the wall face) like `NoSlip`, instead of setting the ghost to 0.
- **sph**: surface tension is now Müller's `−σ·∇²c·n̂/ρ` (was divided by an extra `|n|`) and only acts where `|∇c| > 0.1/h`; coefficients tuned for 1.2 are ~1/h too strong now.
- **sph**: `update_heat` no longer divides by ρ a second time — conduction is ~ρ× faster; use a physical diffusivity (water ≈ 1.4e-7 m²/s).
- **sph**: `update_phase_change` now accumulates latent heat across steps in `fuel` (progress toward the other phase, reset on transition) — don't combine with combustion.
- **sph**: `apply_contact_angle` applies `σ·cos θ/h` toward the wall for wetting (θ < 90°), none at 90°, away for θ > 90°; no longer divides by mass and never acts on degenerate (2D) axes.
- **sph**: PCISPH always runs ≥ 1 iteration and converges on *compression* only.
- **shallow**: continuity flux was halved (waves ran at √(gh/2)); waves now travel at √(gh).
- **shallow**: `breaking_threshold` is now a dimensionless surface slope `|∇η|` (was `|∇η|/depth`, units 1/m).
- **shallow**: `dispersion_coeff` now drives an implicit Peregrine-form filter (`ω² = ghk²/(1+c·h²k²)`, stable for any dt) instead of an explicit 4th-order term that grew exponentially; `use_green_naghdi` selects the depth-weighted operator.
- **shallow**: domain edges are closed walls (no mass leaks into ghost cells).
- **coupling**: drag is quadratic (`½ρ|v|²CdA`, was linear in |v|); added-mass integration applies gravity to the body mass only; FLIP projection is dx-correct; P2G is mass-weighted.
- **mpm**: the `Fluid` stress sign is fixed (compression now pushes back); `DruckerPrager` now has real plasticity with `friction_angle` in **degrees**; `step()` validates every particle's material and volume and returns `Err` instead of producing NaN; grid walls are separating and particles stay in the domain.
- **vortex**: `lamb_oseen_velocity` with `ν·t = 0` returns the point-vortex speed Γ/(2πr) (was 0); `kolmogorov_scale(0, ε)` returns 0 (was ∞).
- **buoyancy**: `reynolds_number` uses |v| and rejects NaN viscosity; `classify_flow` uses |Re|; `terminal_velocity` rejects non-positive ρ, Cd or A individually.

### Changed
- **toolchain**: MSRV 1.89 → 1.99 (edition 2024, latest stable); CI MSRV job, README and CLAUDE.md updated.
- **deps**: hisab 1.1 → 1.4, serde 1.0.229, thiserror 2.0.21, tracing 0.1.44, tracing-subscriber 0.3.23, tracing-chrome 0.7.2, rayon 1.12, serde_json 1.0.154 (dev), criterion 0.5 → 0.8 (dev; benches use `std::hint::black_box`).
- **ai**: the feature no longer pulls `reqwest`, `tokio` (`full`) or `serde_json` — none were used. Lockfile: 214 → 93 packages, no OpenSSL/native-tls.
- **compute**: feature now enables `sph` (it calls the SPH kernels; `--no-default-features --features compute` failed to build).
- **ci**: actions/checkout v4 → v7, codecov-action v4 → v7, action-gh-release v2 → v3.
- **scripts**: `bench-history.sh` now records µs-scale results and names printed on their own line (it silently dropped them — 19 of 57 benchmarks were recorded).
- **deny**: dropped the unused BSD-3-Clause allowance.
- **docs**: docs.rs builds with all features; crate-level module list completed; stale doc claims fixed (Periodic "FFT solve", MacCormack "fallback", k-ε "transport", MPM fluid viscosity).

### Fixed
#### grid
- Projection RHS is now `−dx²·∇·u` for the unit-stencil Poisson system (all four solvers).
- DST and CG solved the opposite-sign system to Gauss-Seidel; all solvers now solve `4p − Σp = rhs` (CG on the SPD operator, relative tolerance).
- Multigrid: residual sign, missing ×4 coarse RHS, and one-sided prolongation fixed; V-cycles now converge, and `pressure_iterations/10` cycles are run.
- MacCormack/BFECC limiter clamped to the destination cell's neighbours (erased features moving > 1 cell/step); now clamps to the backtraced stencil. "BFECC" was MacCormack — it now advects the error-compensated field forward.
- `MacGrid::step`: closed walls, buoyancy only on interior faces (no accumulation on wall faces), Neumann pressure over every cell, density advection (was documented but missing), sizes validated; `cfl_dt` returns `f64::MAX` without flow instead of a length.
- `FluidGrid::step` re-validates sizes (Deserialize/pub fields), rejects NaN `smagorinsky_cs`/`vorticity_confinement`/buoyancy, and errors on non-finite velocity after projection; `cfl_dt` saturates instead of returning ∞.
- `diffuse`, `KEpsilon::step`, `apply_ghost_fluid` ignore mismatched sizes (warning) instead of panicking.
- VOF: closed walls (no leak into the ring), sub-cycled to Courant ≤ 0.5, persistent scratch buffer, slice lengths and non-finite velocity checked.
- Phase field: upwind advection (central + forward Euler is unconditionally unstable), one divergence warning per step (was per cell), wrong-length velocity slices are an error.
#### sph
- DFSPH rewritten in Bender & Koschier order (density solver → advect → divergence solver) with the correct sign of Dρ/Dt, `κ/ρ` scaling, κ computed for all particles first, compression-only correction, and persistent buffers.
- Delta-SPH diffusion had the wrong sign (amplified density noise).
- PCISPH: `max_iterations = 0` teleported particles to the origin; the final integration used one-iteration-old pressures; δ came from an average dominated by sparse particles (now a full-neighbourhood prototype).
- `apply_implicit_viscosity` was K explicit steps; it is now a backward-Euler Gauss-Seidel solve.
- Viscoelastic Oldroyd-B used ∇uᵀ in one term (C_yy drifted in simple shear).
- `use_verlet` was first-order (lagged acceleration, zero first step); it is now velocity Verlet (exact under constant acceleration). New `SphSolver::reset_integrator()` for after reordering.
- Colour-field Laplacian includes the self term; multiphase interface tension uses its own colour field and only acts between phases; out-of-range phases warn once per step; empty `phases` is an error (was a panic).
- `adaptive_dt` with inverted bounds panicked in `f64::clamp`.
- Public kernels return 0 for `h <= 0` (NaN in release); `equation_of_state_tait` handles γ → 0; `create_particle_block` with spacing ≤ 0 returned an endless allocation; `split_particle(.., 1, ..)` moved the centre of mass; `sort_by_zorder` mis-sorted negative coordinates.
- Free functions taking neighbour lists validate them (warning) instead of panicking.
#### shallow
- Riemann path is a proper finite-volume HLL update of (h, hu, hv) with Audusse hydrostatic reconstruction (well-balanced) and mirrored-wall states; it no longer double-counts the pressure gradient and advection. HLL uses dry-bed wave speeds.
- Convective terms are upwinded (centred + forward Euler was unconditionally unstable).
- Breaking detection ignores dry neighbours.
- Sediment: suspended volume `C·h` is conserved through erosion/deposition (bed can no longer rise above the surface); uses `dry_threshold`; validates `dt`/lengths.
- `new()` validates `dx` and height; `step()` validates sizes/gravity/dispersion and errors on non-finite state.
#### coupling
- FLIP pressure solve missing `dx²` (over-corrected ~100× at dx = 0.1).
- `drag_from_particles_indexed` queried only `sample_radius` around the centre and missed bodies larger than the radius; stale hash indices are skipped instead of panicking.
- P2G weights could go negative for particles outside the grid.
- APIC `C` used the quadratic-B-spline factor with bilinear weights; it is now `Σ vᵢ∇wᵢᵀ` (exact for linear fields).
- `flip_ratio` NaN is an error; `particle_level_set` ignores a mismatched buffer.
#### common / compute / misc
- `ParticleArena`: watermark allocations returned stale particles; double free and stale (post-`compact`) handles corrupted the free list or aliased blocks; foreign handles panicked; `alloc(usize::MAX)` overflowed. Handles now carry a generation.
- `ParticleSoa::to_aos`/`write_to_aos` and `PackedParticles::from_soa`/`unpack_*` panicked on mismatched lengths.
- `FluidMaterial::custom`, `FluidConfig::validate`, `check_cfl` accepted NaN/Inf (and negative velocities).
- `KernelDerivatives::dtait_drho` NaN at γ = 0; `dpoly6_dr2(0, 0)` NaN.
- `bridge::heat_to_buoyancy_forcing` NaN at density 0 (ρ cancels).

### Added
- `SphSolver::neighbor_lists()` (the free neighbour-list functions were unusable without it), `SphSolver::reset_integrator()`.
- Constructors for `#[non_exhaustive]` configs that external crates could not build: `HeatConfig::new`, `ViscoelasticConfig::new`, `PhaseChangeConfig::new` / `water_ice`, and `Default` for `SphKernelParams`/`GridKernelParams`/`ShallowKernelParams`.
- `ShallowWater::interior_volume()` (the conserved quantity), `ParticleSoa::is_consistent`/`consistent_len`, `BodyShape::bounding_radius`/`projected_area`.
- **oracle** (Cyrius port tooling, not public API): a hidden, non-default `oracle` feature, kept out of `full`, gives `scripts/oracle/` read-only access to state the public API doesn't expose (port plan M0 step 0). It covers:
  - `oracle_dump()` on every stateful type: `ParticleArena`, `ArenaHandle`, `ShallowWater`, `FluidGrid`, `KEpsilon`, `MacGrid`, `VofField`, `PhaseField`, `FlipSolver`, `MpmSolver` and `SphSolver`. The `SphSolver` dump includes the f32 positions and the neighbour lists in summation order;
  - `oracle::sph_cell_key`;
  - the four grid pressure solvers on a caller's system (`pressure_solve_gauss_seidel`, `_dst`, `_cg`, `_multigrid`);
  - `oracle::invert_3x3`, which is hisab 1.4.0's `matrix_inverse` as the MLS correction calls it.

  The dumps live in child modules (`src/<module>/oracle.rs`), so each audited source gains only a `mod oracle;` declaration. `tests/oracle.rs` (14 tests) runs every solver variant 300 steps plain and 300 steps dumping after each step, and requires bit-identical final states. Its harness was mutation-checked: one extra step in the dumped run fails all 10 solver checks. CI runs it with and without `parallel`. Simulation behaviour is unchanged.

### Changed (port tooling)
- `Cargo.lock` is tracked for the Cyrius port's freeze: the oracle's bits depend on hisab 1.4.0 and glam 0.29.3. This is an exception to the library-crate rule, to be recorded in the port's ADR 0001. The oracle host's glibc is 2.44.

### Security
- `DaimonConfig`'s `Debug` redacts `api_key`.

### Performance
- No speedups are claimed. The full suite (`bench-history.sh`, labels `baseline-1.2.0` → `post-audit-1.3.0`, 57/57 benchmarks) ran on a shared, loaded host and moved ±100% even on unchanged code (`shallow_disturbance`), so it is not used as evidence.
- Pinned A/B (`taskset -c 15`, old/new interleaved, 2 runs each, 2 s measurement) shows parity within run-to-run noise (±20%): `grid_step/63x63` 10.8/13.2 → 13.3/10.5 ms, `grid_step_maccormack/63x63` 12.3/12.9 → 13.8/10.7 ms, `pcisph_step/64` 64.3/68.0 → 80.1/61.5 µs, `solver_step/256` 344/363 → 365/287 µs, `solver_step_st/169` 206/204 → 194/197 µs, `shallow_step/128x128` 546/543 → 567/519 µs, `flip_step/225` 25.0/19.8 → 22.8/28.3 µs.
- Standalone kernels: the new `h > 0` guard first cost ~0.4 ns (+15–30%); reduced to a single comparison, now at parity — `sph_kernels/poly6` 2.35/2.10 → 2.00/2.21 ns, `spiky_grad` 2.11/1.85 → 1.57/1.65 ns, `viscosity_laplacian` 1.53/1.72 → 1.48/1.61 ns.

## [1.2.0]

### Added
- **bridge** — cross-crate primitive-value bridges for pavan (Cp to pressure, dynamic pressure, aero forces, freestream velocity), ushma (temperature to water viscosity, heat to buoyancy forcing, TKE to eddy diffusivity), goonj (velocity to acoustic source strength, TKE to broadband noise, wave height to acoustic pressure)

### Updated
- hisab 1.1.0 -> 1.3.0, zerocopy 0.8.47 -> 0.8.48, iri-string 0.7.11 -> 0.7.12

## [Unreleased-pre-1.2]

### Fixed
- sph: Surface tension div-by-zero in solver step, PCISPH, and multiphase — added threshold floor
- sph: Phase change energy conservation — added `heat_capacity` to `PhaseChangeConfig`, proper J/kg↔K conversion
- sph: Verlet integration used zero prev_accel on every step — fixed resize guard to preserve existing values
- sph: PCISPH delta computed from particle 0 only — now averages over all particles with neighbors
- sph: Implicit viscosity solver ran fixed iterations — added convergence check with early exit
- sph: Contact angle force missing length scale — added smoothing radius divisor
- sph: Non-Newtonian viscosity (Bingham/Herschel-Bulkley) could produce extreme values — added saturation cap
- sph: Multi-phase surface tension early exit used wrong condition — fixed to check `st > 0 || it > 0`
- grid: DST eigenvalue division threshold too small (1e-20 → 1e-12)
- grid: NaN from `sample()` propagated silently — added post-advection divergence check
- grid: Multigrid recursion had no depth limit — capped at 10 levels
- shallow: Riemann solver epsilon too small (1e-10 → 1e-8)
- shallow: HLL flux mixed old/new time levels — now uses consistent snapshot velocities
- shallow: Friction calculation NaN risk near dry threshold — added depth guard
- shallow: Well-balanced pressure gradient used hardcoded 1e-3 — now uses dry threshold
- shallow: Grid minimum size was 1 — raised to 3
- shallow: Sediment array bounds only checked in debug — added runtime check
- coupling: FLIP pressure solver ran fixed 40 iterations — added convergence check with early exit
- coupling: APIC silently degraded to PIC on first step — auto-resize apic_c
- coupling: NaN could propagate through FLIP step — added post-pressure-solve check
- mpm: Deformation gradient could explode — added det(F) clamping
- mpm: Particle/mpm_data count mismatch silently truncated — now returns error
- common: `ParticleArena::active_count` could underflow — switched to `saturating_sub`
- vortex: `lamb_oseen_velocity` div-by-zero when viscosity=0 — added guard
- buoyancy: `reynolds_number` and `terminal_velocity` accepted NaN/Inf inputs — added validation

### Added
- mpm: `ConstitutiveModel::validate()` — validates material parameters (Young's modulus, Poisson ratio, bulk modulus)
- grid: `GridConfig` validation — smagorinsky_cs, vorticity_confinement, iteration counts
- grid: BFECC advection wired up via `use_bfecc` flag
- vof: `is_surface_with_threshold()`, `is_full_with_threshold()`, `is_empty_with_threshold()` — configurable threshold
- grid: Conjugate gradient pressure solver via hisab (`GridConfig.use_cg`) — matrix-free, SPD-optimized
- coupling: `drag_from_particles_indexed()` — spatial-hash-accelerated drag using hisab `SpatialHash`

### Changed
- sph: MLS 3×3 matrix inversion now delegates to `hisab::num::matrix_inverse` (LU-based)
- sph: Density threshold standardized via `MIN_DENSITY` constant across all solvers
- sph: `MultiPhaseConfig::get()` now logs warning on invalid phase index fallback
- grid: All pressure solvers now return `Result<()>` consistently
- grid: k-epsilon solver uses scratch buffer swap instead of cloning
- vortex: `kolmogorov_scale` uses log-form for numeric stability
- phase_field: Phase field clamping now warns via tracing when |φ| > 1.5
- shallow: CFL condition accounts for dispersive stability limit

## [1.1.0] - 2026-03-26

### Added
- sph: Wendland C2/C4 kernels (`kernel_wendland_c2`, `kernel_wendland_c4`) — no tensile instability
- sph: Delta-SPH density diffusion (`apply_delta_sph`) — pressure noise reduction
- sph: Velocity Verlet integration (`SphSolver.use_verlet`) — 2nd-order symplectic
- sph: Non-Newtonian viscosity (`NonNewtonianViscosity`) — power-law, Bingham, Herschel-Bulkley
- sph: Z-order particle sorting (`sort_by_zorder`) — Morton code for cache-friendly access
- sph: Enhanced `adaptive_dt` with force and viscous CFL constraints
- grid: `FluidGrid::cfl_dt()` — advective + viscous CFL timestep
- grid: Smagorinsky SGS turbulence model (`GridConfig.smagorinsky_cs`)
- sph: `ReactionProvider` trait for pluggable chemistry backends (kimiya, etc.)
- sph: `update_reaction()` — generic reaction step using any `ReactionProvider`
- sph: `CombustionConfig` implements `ReactionProvider` for backward compatibility
- sph: DFSPH solver (`step_dfsph`) — divergence-free SPH, 2-stage correction
- sph: MLS gradient correction (`compute_gradient_corrections`, `apply_gradient_correction`)
- grid: Multigrid V-cycle pressure solver (`GridConfig.use_multigrid`)
- shallow: HLL Riemann solver for flux computation (`ShallowWater.use_riemann`)
- coupling: APIC transfers (`FlipSolver.use_apic`) — affine particle-in-cell, angular momentum conserving
- All vector types migrated from `[f64; 3]` to `hisab::DVec3` (SIMD-accelerated)
- grid: Staggered MAC grid (`MacGrid`) — velocity at faces, pressure at centers, no checkerboard
- grid: Ghost fluid method (`apply_ghost_fluid`) — pressure jump at multi-phase interfaces
- vof: Volume of Fluid (`VofField`) — free surface tracking with donor-acceptor advection
- sph: Contact angle / wetting (`apply_contact_angle`) — Young's equation at solid boundaries
- sph: Phase change (`PhaseChangeConfig`, `update_phase_change`) — melting, solidification, evaporation
- shallow: Sediment transport (`SedimentConfig`, `update_sediment`) — Shields criterion, erosion/deposition
- mpm: Material Point Method (`MpmSolver`, `MpmParticle`) — neo-Hookean, fluid, Drucker-Prager constitutive models
- sph: Tait equation of state (`equation_of_state_tait`) for compressible flow
- sph: Implicit viscosity solver (`apply_implicit_viscosity`) — removes viscous CFL restriction
- sph: Particle splitting (`split_particle`) and merging (`merge_particles`) with mass/momentum conservation
- sph: Batch kernel evaluation (`batch_poly6`, `batch_distance_squared`) — SIMD-friendly patterns
- coupling: Narrow-band FLIP option (`FlipSolver.narrow_band_cells`)
- compute: `NeuralCorrector` trait for learned simulation correction (FNO, GNS backends)
- compute: `KernelDerivatives` — analytical Jacobians for differentiable simulation (dPoly6/dr², dEOS/dρ, dTait/dρ)
- grid: k-epsilon turbulence model (`KEpsilon`) — two-equation RANS with production/dissipation transport
- grid: BFECC advection (`advect_bfecc`) — 2nd-order error-compensating advection
- shallow: Green-Naghdi fully nonlinear dispersion (`use_green_naghdi`)
- sph: Foam/spray/bubble generation (`SecondaryParticle`, `generate_secondary_particles`, `update_secondary_particles`)
- compute: `KernelDerivatives::autodiff_gradient()` — reverse-mode autodiff via hisab's `Tape`/`Var`
- hisab dependency updated to 1.1 (reverse-mode autodiff, stiff ODE solvers, SDFs, k-epsilon eigendecomposition)
- shallow: `ShallowWater::cfl_dt()` — wave speed CFL timestep

### Previously Added
- shallow: Non-linear shallow water equations with convective acceleration (u·∇u)
- shallow: Flux-form continuity equation (∂(hu)/∂x + ∂(hv)/∂y) for proper mass transport
- shallow: Neumann boundary enforcement (zero-gradient height, reflected normal velocity)
- shallow: Manning's bed friction (`manning_n` per-cell roughness) with implicit treatment for stability
- shallow: Wetting/drying transitions (`dry_threshold`, `is_wet()`) with velocity zeroing, depth clamping
- shallow: Wave breaking detection (`breaking_threshold`, `is_breaking()`) with turbulent bore dissipation
- shallow: Boussinesq dispersive correction (`dispersion_coeff`) for short-wavelength accuracy
- shallow: Well-balanced hydrostatic reconstruction for terrain-following pressure gradient (lake-at-rest preserved over sloped bathymetry)
- sph: Rayon parallelism for SphSolver density and force loops (`parallel` feature gate)
- grid: Rayon parallelism for advection and pressure projection (`parallel` feature gate)
- common: `ParticleSoa` — SIMD-friendly Structure-of-Arrays layout with AOS↔SOA conversion
- compute: GPU-agnostic `ComputeBackend` trait, `PackedParticles` f32 buffer packing, kernel parameter structs for SPH/grid/shallow
- common: `ParticleArena` — pre-allocated pool with alloc/free/compact, `ArenaHandle` for block references
- logging: `init_profiling()` — chrome://tracing JSON output for flame graph analysis via tracing-chrome
- sph: Multi-phase SPH (`step_multiphase`, `MultiPhaseConfig`, `PhaseProperties`) with per-phase EOS, viscosity, and interface tension
- common: `FluidParticle.phase` field (u8) for multi-phase particle identification
- phase_field: `PhaseField` — Allen-Cahn interface tracking with advection, double-well potential, Neumann BCs
- sph: Viscoelastic fluids (`ViscoelasticConfig`, `update_viscoelastic`) — Oldroyd-B conformation tensor evolution with elastic stress
- sph: Heat transfer (`HeatConfig`, `update_heat`) — SPH Laplacian-based conduction with implicit advection
- common: `FluidParticle.temperature` field (default 293.15 K)
- sph: Combustion reaction (`CombustionConfig`, `update_combustion`) — fuel depletion with exothermic heat release
- common: `FluidParticle.fuel` field for combustion concentration

### Changed
- `#[non_exhaustive]` added to 18 public structs for forward-compatible API evolution
- Cargo.toml: license corrected from deprecated `GPL-3.0` to `GPL-3.0-only`
- shallow: NaN dt now rejected (consistent with grid.rs)
- sph: `SpatialHash::new()` error properly propagated (hisab API change)
- sph: Aligned standalone `pressure_force()` and brute-force `step()` with symmetric formula
- grid: Persistent scratch buffers (zero allocation after first step)
- grid: DST pressure solver now propagates transform errors instead of silent ignore
- common: `FluidConfig::validate()` rejects NaN/Inf for dt, smoothing_radius, density, gas_constant
- common: `FluidConfig::validate()` rejects inverted bounds (min > max)
- sph: PCISPH divergence detection (NaN/Inf check on acceleration and predicted positions)
- sph: Aligned standalone `pressure_force()` and brute-force `step()` with symmetric formula
- coupling: FLIP solver particle boundary clamping to grid domain
- coupling: Directional drag cross-section for box shapes (projected onto velocity normal)
- deny.toml for cargo-deny license and advisory checking

## [0.24.3] - 2026-03-24

### Added
- sph: PCISPH pressure solver (`step_pcisph`) with iterative density-error correction
- sph: Adaptive timestep (`SphSolver::adaptive_dt`) — CFL-driven with NaN guards
- grid: MacCormack advection (forward-backward correction with neighbor clamping)
- grid: Periodic boundary conditions (wrapping edges, periodic sampling, `advect_periodic`)
- grid: DST Poisson solver (exact solve via hisab `dst`/`idst` for wall boundaries)
- coupling: `RigidBody` with `BodyShape` (sphere, box) signed distance functions
- coupling: Two-way SPH-rigid body coupling (`couple_sph_bodies`) with penalty forces
- coupling: `FlipSolver` — FLIP/PIC hybrid (P2G, pressure projection, G2P blend)
- coupling: Particle-level set surface tracking (`particle_level_set`)
- coupling: Added mass computation (`effective_mass`, `integrate_bodies_with_added_mass`)
- coupling: Drag from velocity fields (`drag_from_particles`)

### Changed
- grid: Pressure projection uses DST for wall BCs (exact), GS for periodic
- grid: Correct dx scaling in advection, diffusion, divergence, projection
- grid: Diffusion uses correct implicit equation with preserved RHS
- grid: Warm-start pressure solve from previous step
- grid: Minimum grid size 4x4 enforced
- grid: `GridConfig` is `#[non_exhaustive]` with `Serialize`/`Deserialize`/`Default`
- grid: NaN propagation in `sample()`, NaN/Inf dt validation
- coupling: Central-difference surface normals with degenerate fallback
- coupling: Correct dt scaling in FLIP pressure projection
- coupling: `FlipSolver` grid dims are private with getters
- hisab: Switched to local path dependency for DST access

## [0.2.0] - 2026-03-24

### Added
- sph: `SphSolver` — stateful solver with hisab `SpatialHash` for O(n·k) neighbor queries
- sph: Neighbor list caching via `query_cell` (zero per-particle allocation)
- sph: Surface tension via CSF model with poly6 gradient/laplacian kernels
- docs: Roadmap at `docs/development/roadmap.md`

### Changed
- sph: Symmetric pressure formula (`p_i/ρ_i² + p_j/ρ_j²`) for momentum conservation
- sph: Persistent scratch buffers — zero per-step allocations after first call
- common: `FluidParticle` and `FluidConfig` derive `Copy`
- common: Added `speed_squared()`, `distance_squared_to()`
- common: `FluidConfig::validate()` checks gas_constant, boundary_damping
- common: `FluidMaterial::custom()` validates speed_of_sound
- grid: `FluidGrid::new()` validates dx, checks overflow
- grid: `max_speed()` avoids per-cell sqrt
- buoyancy: `terminal_velocity` and `reynolds_number` return `Result`
- buoyancy: `FlowRegime` is `#[non_exhaustive]`
- error: `Diverged` and `InvalidParameter` use `Cow<'static, str>`

### Fixed
- sph: NaN/Inf divergence detection folded into force loop
- sph: `create_particle_block` pre-allocates with `Vec::with_capacity`
- shallow: Scratch buffers resize on deserialization to prevent panic

## [0.1.0] - 2026-03-23

### Added
- common: FluidParticle, FluidMaterial (water/oil/honey/air/lava), FluidConfig with CFL validation
- sph: Poly6/Spiky/Viscosity kernels, density/pressure, forces, particle stepping, block creation
- grid: FluidGrid with velocity/pressure/density fields, Gauss-Seidel diffusion, kinetic energy
- shallow: ShallowWater heightfield, circular disturbances, wave propagation, volume conservation
- buoyancy: Archimedes buoyancy, drag force, terminal velocity, Reynolds number, flow regime
- vortex: 2D vorticity, Lamb-Oseen/Rankine vortex models, enstrophy, Kolmogorov microscale
- error: PravashError with #[non_exhaustive], CFL violation detection
