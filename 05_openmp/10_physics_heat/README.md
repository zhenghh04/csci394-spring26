# Physics kernel: 2D heat diffusion

Finite-difference stencil update (5-point) for a 2D heat equation.

![5-point stencil schematic](stencil_heat.png)

Physical model (continuous):
$$
\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)
$$
We use an explicit 5-point stencil for the Laplacian:
$$
u_{i,j}^{t+1} = u_{i,j}^t + \alpha \left(u_{i-1,j}^t + u_{i+1,j}^t + u_{i,j-1}^t + u_{i,j+1}^t - 4u_{i,j}^t\right)
$$

Algorithm (what the code does):
- Initialize a 2D grid with a hot spot at the center.
- For each time step, update every interior cell using the 5-point stencil.
- Swap the old and new grids and repeat for a fixed number of steps.

## Build
```bash
make
```

## Run
```bash
OMP_NUM_THREADS=4 ./heat2d
```

## Assignment
- Assignment materials were moved to:
  - `../assignments/ASSIGNMENT_loopy_parallel.md`
  - `../assignments/loopy_assignment.c`
