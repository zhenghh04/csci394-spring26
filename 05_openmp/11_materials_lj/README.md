# Materials kernel: Lennard-Jones forces

Computes pairwise Lennard-Jones forces for particles in 2D.

![LJ pair distance schematic](lj_pairs.png)

Physical model:
$$
U(r) = 4\varepsilon \left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]
$$
$$
F(r) = -\nabla U(r) = 24\varepsilon\left(2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right)\frac{1}{r^2}\,\vec{r}
$$
This kernel computes the force on each particle by summing contributions
from all other particles within a cutoff radius.

Algorithm (what the code does):
- Place particles on a simple 2D grid.
- For each particle `i`, loop over all other particles `j`.
- Compute distance `r` and skip if `r` is beyond the cutoff.
- Accumulate the Lennard-Jones force contribution into `fx[i], fy[i]`.

## Build
```bash
make
```

## Run
```bash
OMP_NUM_THREADS=4 ./lj_forces
```
