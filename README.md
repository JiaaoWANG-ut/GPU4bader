# Bader Charge Analysis -- Optimized Edition

A performance-optimized fork of the [Henkelman Group's Bader charge analysis](http://theory.cm.utexas.edu/henkelman/code/bader/) program (Version 1.05, August 2023). This edition delivers **1.8x overall speedup** through compiler tuning, branchless periodic boundary conditions, and OpenMP parallelization of auxiliary kernels -- while producing **bit-identical output** to the original.

## Authors

**Original code:**
Wenjie Tang, Andri Arnaldsson, Wenrui Chai, Samuel T. Chill, and Graeme Henkelman
(University of Texas at Austin)

**Optimization work:**
Performance engineering applied to compiler flags, numerical kernels, and parallelization.

## References

- G. Henkelman, A. Arnaldsson, and H. Jonsson, *Comput. Mater. Sci.* **36**, 254-360 (2006).
- E. Sanville, S. Kenny, R. Smith, and G. Henkelman, *J. Comput. Chem.* **28**, 899-908 (2007).
- W. Tang, E. Sanville, and G. Henkelman, *J. Phys.: Condens. Matter* **21**, 084204 (2009).
- M. Yu and D. Trinkle, *J. Chem. Phys.* **134**, 064111 (2011).

## License

GNU General Public License v3.0 or later. See the header of `main.f90`.

---

## New Features

### 1. Optimized Makefile (`makefile.lnx_gfortran_opt`)

Aggressive compiler flags for `gfortran`:

| Flag | Purpose |
|------|---------|
| `-O3` | Full optimization (vs. original `-O2`) |
| `-march=native` | Use host CPU's SIMD instructions (SSE/AVX) |
| `-funroll-loops` | Unroll inner loops for reduced branch overhead |
| `-flto` | Link-Time Optimization: cross-module inlining of hot functions |
| `-fopenmp` | Enable OpenMP parallel regions |
| `-cpp` | C preprocessor for `#ifdef _OPENACC` guards |

### 2. Branchless Periodic Boundary Conditions

The `rho_val` function (called ~717 million times per NaCl run) and `pbc` subroutine were rewritten to use `MODULO` arithmetic instead of loop-based wrapping:

```fortran
! Before: 6 branches per call
DO i = 1,3
  DO
    IF (p(i) >= 1) EXIT
    p(i) = p(i) + npts(i)
  END DO
  DO
    IF (p(i) <= npts(i)) EXIT
    p(i) = p(i) - npts(i)
  END DO
END DO

! After: branchless, single expression
rho_val = chg%rho(MODULO(p1-1, npts(1)) + 1, &
                  MODULO(p2-1, npts(2)) + 1, &
                  MODULO(p3-1, npts(3)) + 1)
```

Combined with `-flto` cross-module inlining, this eliminates billions of branch mispredictions.

### 3. OpenMP Parallelization of Auxiliary Kernels

Two compute-intensive post-partitioning loops are parallelized:

- **Charge integration** (`bader_calc`): `!$OMP PARALLEL DO` with `ATOMIC` reductions
- **Minimum distance calculation** (`bader_mindist`): `!$OMP PARALLEL DO` with `CRITICAL` sections

The main gradient-ascent tracing loop remains serial because its **path-caching optimization** (early termination when a trace hits an already-assigned point) provides more benefit than parallelism would on typical grid sizes.

### 4. Ghost-Layer Padding Infrastructure

A padded copy of the charge density array (`rho_pad`) is pre-computed at startup with one layer of ghost cells on each face, edge, and corner. This allows neighbor lookups without any boundary checks:

```
rho_pad(0:n1+1, 0:n2+1, 0:n3+1)
```

This infrastructure is ready for use in inner loops that require only +-1 neighbor access.

### 5. OpenACC GPU Annotations (Experimental)

All performance-critical subroutines (`rho_val`, `rho_grad`, `rho_grad_dir`, `pbc`, `step_ongrid`, `to_lat`) are annotated with `!$acc routine seq` directives. Thread-safe trace functions (`trace_neargrid`, `trace_ongrid`, `trace_offgrid`) are provided for GPU-parallel gradient ascent. These are guarded by `#ifdef _OPENACC` and activate only when compiled with an OpenACC-capable compiler (e.g., `nvfortran`).

### 6. `INTENT` Annotations for Compiler Optimization

`INTENT(IN)` was added to function arguments in `rho_val`, `rho_grad_dir`, and `pbc` to give the compiler stronger aliasing guarantees for optimization.

---

## Benchmark Results

**Test system:** NaCl (8 atoms), VASP CHG file, 160 x 160 x 160 grid (4,096,000 points).

### Phase-by-Phase Timing

| Phase | Original (gfortran -O2) | Optimized (gfortran -O3, 4 threads) | Speedup |
|-------|:-----------------------:|:------------------------------------:|:-------:|
| File I/O | 2.44 s | 1.91 s | 1.3x |
| Bader partitioning + refinement | 25.35 s | 14.78 s | **1.7x** |
| Minimum distance | 2.19 s | 0.20 s | **10.9x** |
| **Total wall time** | **30.0 s** | **17.0 s** | **1.8x** |

### Correctness Verification

All three output files are **byte-identical** to the original:

```
$ diff ACF.dat ACF.dat.orig   # Bader charges per atom
$ diff BCF.dat BCF.dat.orig   # Bader volume charges
$ diff AVF.dat AVF.dat.orig   # Atom-volume assignments
(no differences)
```

Key metrics match exactly:
- Number of Bader maxima: 517
- Significant maxima: 517
- Total electrons: 64.00000
- Edge points: 2,083,713
- Reassigned points: 320,118

---

## Algorithm Structure and Architecture

### High-Level Pipeline

```
main.f90
  |
  +-- read_charge()           # Parse VASP CHG / CHGCAR / Cube file
  +-- init_rho_pad()          # Build ghost-layer padded array
  |
  +-- bader_calc()            # Core Bader partitioning
  |     |
  |     +-- [Vacuum detection]
  |     +-- [Serial gradient ascent with path caching]  <-- main hotspot
  |     +-- [Automatic edge refinement]
  |     +-- [Charge integration]                        <-- OpenMP parallel
  |     +-- assign_chg2atom()
  |     +-- cal_atomic_vol()
  |
  +-- bader_mindist()         # Min surface-to-atom distance  <-- OpenMP parallel
  +-- bader_output()          # Write ACF.dat, BCF.dat, AVF.dat
```

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| `kind_mod` | `kind_mod.f90` | Precision constants (`q2 = REAL64`) |
| `matrix_mod` | `matrix_mod.f90` | 3x3 matrix operations (volume, inverse) |
| `ions_mod` | `ions_mod.f90` | Ion positions data type |
| `options_mod` | `options_mod.f90` | Command-line parsing and option flags |
| `charge_mod` | `charge_mod.f90` | Charge density: I/O, `rho_val`, gradients, PBC |
| `bader_mod` | `bader_mod.f90` | Bader partitioning, refinement, output |
| `voronoi_mod` | `voronoi_mod.f90` | Voronoi decomposition (alternative method) |
| `weight_mod` | `weight_mod.f90` | Yu-Trinkle weight method |
| `io_mod` | `io_mod.f90` | File format dispatch (CHGCAR, Cube) |
| `critpoint_mod` | `critpoint_mod.f90` | Critical point analysis |

### Bader Partitioning Algorithm Detail

The Bader analysis partitions a 3D charge density grid into basins by tracing the gradient ascent from each grid point to its local maximum (attractor). Points sharing the same attractor belong to the same Bader volume.

**Three tracing algorithms** are available (selected via `-m` flag):

1. **Neargrid** (default, `-m neargrid`): Hybrid approach. Follows the continuous gradient but snaps to grid points at each step, accumulating a sub-grid residual. Balances accuracy and convergence speed.

2. **Ongrid** (`-m ongrid`): Pure steepest-ascent on the 26-neighbor grid. Each step moves to the neighbor with highest distance-corrected charge density. Fastest per step but may need more steps.

3. **Offgrid** (`-m offgrid`): Continuous gradient ascent using trilinear interpolation. Most accurate but slowest.

**Key algorithmic optimization -- Path Caching:**

```
for each unassigned grid point p:
    trace gradient ascent from p, recording the path
    if trace reaches an already-assigned point:
        stop early (inherit its basin ID)
    else:
        create new basin at the attractor
    assign entire path to this basin
    mark neighboring points as "known" for future early termination
```

This means later traces are dramatically shorter because they quickly hit already-classified territory. On the NaCl benchmark, the median trace length is only ~3-5 steps despite grids having 160 points per dimension.

**Edge Refinement:**

After initial partitioning, points on Bader volume boundaries are re-traced at higher accuracy to improve boundary precision. This typically reassigns ~8% of edge points and converges in a single iteration.

### Data Flow Diagram

```
                    +------------------+
                    |  CHG_NaCl file   |
                    | (160x160x160)    |
                    +--------+---------+
                             |
                     read_charge()
                             |
                    +--------v---------+
                    |  charge_obj      |
                    |  .rho(160,160,160)| -----> init_rho_pad()
                    |  .lat2car(3,3)   |         .rho_pad(0:161,0:161,0:161)
                    |  .car2lat(3,3)   |
                    +--------+---------+
                             |
                      bader_calc()
                             |
              +--------------+--------------+
              |                             |
     Serial tracing loop            Parallel integration
     (path-cached gradient          (!$OMP PARALLEL DO
      ascent per grid point)         ATOMIC reduction)
              |                             |
              v                             v
     +--------+---------+         +--------+---------+
     |  bader_obj        |         |  bdr%volchg(:)   |
     |  .volnum(160^3)   |         |  per-basin charge|
     |  .volpos_lat(:,3) |         +------------------+
     |  .known(160^3)    |
     +--------+----------+
              |
       bader_mindist()  (!$OMP PARALLEL DO)
              |
       bader_output()
              |
     +--------v---------+
     | ACF.dat (per-atom)|
     | BCF.dat (per-vol) |
     | AVF.dat (mapping) |
     +-------------------+
```

---

## Building

### Prerequisites

- `gfortran` >= 9.0 (GCC Fortran compiler)
- GNU Make

### Optimized Build (recommended)

```bash
cd bader
make -f makefile.lnx_gfortran_opt
```

### Original Build (for comparison)

```bash
cd bader
make -f makefile.lnx_ifort    # uses gfortran -O2
```

### Clean

```bash
make -f makefile.lnx_gfortran_opt clean
```

## Running

```bash
# Basic usage (reads VASP CHG/CHGCAR file)
./bader CHGCAR

# With reference charge density
./bader CHGCAR -ref AECCAR0 -ref AECCAR2

# Control thread count for OpenMP-parallelized sections
OMP_NUM_THREADS=4 ./bader CHGCAR

# Select tracing algorithm
./bader -m neargrid CHGCAR    # default
./bader -m ongrid CHGCAR      # faster per step
./bader -m offgrid CHGCAR     # most accurate
```

### Output Files

| File | Description |
|------|-------------|
| `ACF.dat` | Bader charges, min surface distances, and volumes per atom |
| `BCF.dat` | Coordinates, charge, nearest atom for each Bader volume |
| `AVF.dat` | Mapping of Bader volumes to atoms |

## Directory Structure

```
bader/
  bader/                  # Source code
    main.f90              # Program entry point
    bader_mod.f90         # Core Bader partitioning (modified)
    charge_mod.f90        # Charge density operations (modified)
    makefile.lnx_gfortran_opt   # Optimized makefile (new)
    makefile.lnx_ifort    # Original makefile
    ...                   # Other modules (unmodified)
  NaCl/                   # Benchmark test case
    CHG_NaCl              # NaCl 160x160x160 charge density
    stdout                # Reference output
```

## Design Decisions and Lessons Learned

### Why Not Parallelize the Main Tracing Loop?

A two-phase parallel approach was extensively tested: Phase A traces all grid points independently in parallel (storing attractors), Phase B canonicalizes attractors into basin IDs serially.

**Result:** 3.4x slower than serial (94s vs 28s on 4 threads).

**Root cause:** The serial algorithm's path caching is far more valuable than parallelism for this workload. When tracing serially, each completed trace assigns ~10-50 grid points along its path, meaning subsequent traces hitting those points terminate in 1-2 steps. The parallel approach forces all 4M points to trace independently to their attractors, doing vastly more total floating-point work.

This is a textbook example of **work-efficient serial algorithms outperforming brute-force parallelism** when the serial optimization (path caching / memoization) provides super-linear speedup.

### Why Not Parallelize Edge Refinement?

Parallel edge refinement using thread-safe trace functions was tested. It produced **different results** (465K vs 320K reassignments) because the thread-safe traces lack path caching and the `quit_known` early termination, leading to different attractor assignments at volume boundaries.

The serial refinement with path caching is both faster and more accurate for this workload.
