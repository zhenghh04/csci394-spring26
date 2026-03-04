# 05 Collectives

Separate demos for core MPI collectives:
- `bcast` -> `MPI_Bcast`
- `scatter` -> `MPI_Scatter`
- `gather` -> `MPI_Gather`
- `reduce` -> `MPI_Reduce`
- `reduce_scatter` -> `MPI_Reduce_scatter`

## `reduce_scatter` schematic (4 ranks, `recvcounts=[1,1,1,1]`)
```text
Each rank contributes a length-4 vector:
  R0: [10, 11, 12, 13]
  R1: [20, 21, 22, 23]
  R2: [30, 31, 32, 33]
  R3: [40, 41, 42, 43]

Element-wise reduce (SUM):
  [10+20+30+40, 11+21+31+41, 12+22+32+42, 13+23+33+43]
  = [100, 104, 108, 112]

Scatter one reduced element to each rank:
  R0 <- 100
  R1 <- 104
  R2 <- 108
  R3 <- 112
```

## FSDP communication schematic (`all_gather` + `reduce_scatter`)
Example with 4 ranks and parameter vector `W=[w0..w7]` sharded by 2 elements/rank.

Legend:
- <span style="background-color:#fde68a;padding:2px 6px;border-radius:4px;">R0</span>
- <span style="background-color:#bfdbfe;padding:2px 6px;border-radius:4px;">R1</span>
- <span style="background-color:#fecaca;padding:2px 6px;border-radius:4px;">R2</span>
- <span style="background-color:#bbf7d0;padding:2px 6px;border-radius:4px;">R3</span>

Initial parameter shards (persistent):
- <span style="background-color:#fde68a;padding:2px 6px;border-radius:4px;">R0</span> `[w0 w1]`
- <span style="background-color:#bfdbfe;padding:2px 6px;border-radius:4px;">R1</span> `[w2 w3]`
- <span style="background-color:#fecaca;padding:2px 6px;border-radius:4px;">R2</span> `[w4 w5]`
- <span style="background-color:#bbf7d0;padding:2px 6px;border-radius:4px;">R3</span> `[w6 w7]`

Forward pass (`all_gather` parameters):

<table>
  <tr>
    <th style="padding:4px 8px;">Full W on each rank</th>
    <td style="background:#fde68a;padding:6px 10px;text-align:center;">w0</td>
    <td style="background:#fde68a;padding:6px 10px;text-align:center;">w1</td>
    <td style="background:#bfdbfe;padding:6px 10px;text-align:center;">w2</td>
    <td style="background:#bfdbfe;padding:6px 10px;text-align:center;">w3</td>
    <td style="background:#fecaca;padding:6px 10px;text-align:center;">w4</td>
    <td style="background:#fecaca;padding:6px 10px;text-align:center;">w5</td>
    <td style="background:#bbf7d0;padding:6px 10px;text-align:center;">w6</td>
    <td style="background:#bbf7d0;padding:6px 10px;text-align:center;">w7</td>
  </tr>
</table>

Backward pass (`reduce_scatter` gradients):
1. Each rank computes local full gradient contribution `dW_local=[g0..g7]`.
2. Gradients are summed across ranks and scattered back as shards:
- <span style="background-color:#fde68a;padding:2px 6px;border-radius:4px;">R0</span> gets reduced shard `[G0 G1]`
- <span style="background-color:#bfdbfe;padding:2px 6px;border-radius:4px;">R1</span> gets reduced shard `[G2 G3]`
- <span style="background-color:#fecaca;padding:2px 6px;border-radius:4px;">R2</span> gets reduced shard `[G4 G5]`
- <span style="background-color:#bbf7d0;padding:2px 6px;border-radius:4px;">R3</span> gets reduced shard `[G6 G7]`

Net effect in FSDP:
- `all_gather` makes full params temporary for compute.
- `reduce_scatter` keeps grads sharded for optimizer update and memory savings.

## `scatter -> local compute -> gather` schematic (distributed matvec sketch)
```c
// Root has x_global[N], scatters x blocks
MPI_Scatter(x_global, n_local, MPI_DOUBLE, x_local, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
// Each rank computes local y_local = A_local * x_local (or local contribution)
local_matvec(A_local, x_local, y_local, n_local);
// Gather local outputs to root for global vector y
MPI_Gather(y_local, n_local, MPI_DOUBLE, y_global, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

Example with 4 ranks, `N=8`, `n_local=2`

Legend:
- <span style="background-color:#fde68a;padding:2px 6px;border-radius:4px;">R0</span>
- <span style="background-color:#bfdbfe;padding:2px 6px;border-radius:4px;">R1</span>
- <span style="background-color:#fecaca;padding:2px 6px;border-radius:4px;">R2</span>
- <span style="background-color:#bbf7d0;padding:2px 6px;border-radius:4px;">R3</span>

`A * x = y` view (columns of `A` match scattered `x` ownership):

<table>
  <tr>
    <th style="padding:4px 8px;">A columns</th>
    <th style="background:#fde68a;padding:4px 8px;">c0</th>
    <th style="background:#fde68a;padding:4px 8px;">c1</th>
    <th style="background:#bfdbfe;padding:4px 8px;">c2</th>
    <th style="background:#bfdbfe;padding:4px 8px;">c3</th>
    <th style="background:#fecaca;padding:4px 8px;">c4</th>
    <th style="background:#fecaca;padding:4px 8px;">c5</th>
    <th style="background:#bbf7d0;padding:4px 8px;">c6</th>
    <th style="background:#bbf7d0;padding:4px 8px;">c7</th>
  </tr>
  <tr>
    <th style="padding:4px 8px;">x</th>
    <td style="background:#fde68a;text-align:center;">x0</td>
    <td style="background:#fde68a;text-align:center;">x1</td>
    <td style="background:#bfdbfe;text-align:center;">x2</td>
    <td style="background:#bfdbfe;text-align:center;">x3</td>
    <td style="background:#fecaca;text-align:center;">x4</td>
    <td style="background:#fecaca;text-align:center;">x5</td>
    <td style="background:#bbf7d0;text-align:center;">x6</td>
    <td style="background:#bbf7d0;text-align:center;">x7</td>
  </tr>
</table>

Interpretation:
- <span style="background-color:#fde68a;padding:2px 6px;border-radius:4px;">R0</span> owns columns `c0:c1` and multiplies by `[x0 x1]`
- <span style="background-color:#bfdbfe;padding:2px 6px;border-radius:4px;">R1</span> owns columns `c2:c3` and multiplies by `[x2 x3]`
- <span style="background-color:#fecaca;padding:2px 6px;border-radius:4px;">R2</span> owns columns `c4:c5` and multiplies by `[x4 x5]`
- <span style="background-color:#bbf7d0;padding:2px 6px;border-radius:4px;">R3</span> owns columns `c6:c7` and multiplies by `[x6 x7]`
- Each rank computes a local contribution to `y`, then root gathers/combines as your program design requires.

Root `x_global` (block-partitioned by rank):

<table>
  <tr>
    <td style="background:#fde68a;padding:6px 10px;text-align:center;">x0</td>
    <td style="background:#fde68a;padding:6px 10px;text-align:center;">x1</td>
    <td style="background:#bfdbfe;padding:6px 10px;text-align:center;">x2</td>
    <td style="background:#bfdbfe;padding:6px 10px;text-align:center;">x3</td>
    <td style="background:#fecaca;padding:6px 10px;text-align:center;">x4</td>
    <td style="background:#fecaca;padding:6px 10px;text-align:center;">x5</td>
    <td style="background:#bbf7d0;padding:6px 10px;text-align:center;">x6</td>
    <td style="background:#bbf7d0;padding:6px 10px;text-align:center;">x7</td>
  </tr>
</table>

After `MPI_Scatter`:
- <span style="background-color:#fde68a;padding:2px 6px;border-radius:4px;">R0</span> `x_local=[x0 x1]`
- <span style="background-color:#bfdbfe;padding:2px 6px;border-radius:4px;">R1</span> `x_local=[x2 x3]`
- <span style="background-color:#fecaca;padding:2px 6px;border-radius:4px;">R2</span> `x_local=[x4 x5]`
- <span style="background-color:#bbf7d0;padding:2px 6px;border-radius:4px;">R3</span> `x_local=[x6 x7]`

Local compute on each rank:
- <span style="background-color:#fde68a;padding:2px 6px;border-radius:4px;">R0</span> `y_local = A_local * x_local -> [y0 y1]`
- <span style="background-color:#bfdbfe;padding:2px 6px;border-radius:4px;">R1</span> `y_local = A_local * x_local -> [y2 y3]`
- <span style="background-color:#fecaca;padding:2px 6px;border-radius:4px;">R2</span> `y_local = A_local * x_local -> [y4 y5]`
- <span style="background-color:#bbf7d0;padding:2px 6px;border-radius:4px;">R3</span> `y_local = A_local * x_local -> [y6 y7]`

After `MPI_Gather` to root (`y_global`):

<table>
  <tr>
    <td style="background:#fde68a;padding:6px 10px;text-align:center;">y0</td>
    <td style="background:#fde68a;padding:6px 10px;text-align:center;">y1</td>
    <td style="background:#bfdbfe;padding:6px 10px;text-align:center;">y2</td>
    <td style="background:#bfdbfe;padding:6px 10px;text-align:center;">y3</td>
    <td style="background:#fecaca;padding:6px 10px;text-align:center;">y4</td>
    <td style="background:#fecaca;padding:6px 10px;text-align:center;">y5</td>
    <td style="background:#bbf7d0;padding:6px 10px;text-align:center;">y6</td>
    <td style="background:#bbf7d0;padding:6px 10px;text-align:center;">y7</td>
  </tr>
</table>

## Build
```bash
make
```

## Run
```bash
mpiexec -n 4 ./bcast
mpiexec -n 4 ./scatter
mpiexec -n 4 ./gather
mpiexec -n 4 ./reduce
mpiexec -n 4 ./reduce_scatter
```
