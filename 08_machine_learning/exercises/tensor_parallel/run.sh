module load conda
conda activate
mpiexec -np 2 --ppn 2  ./launcher.sh python3 -u matmul_tensor_parallel.py
mpiexec -np 4 --ppn 4  ./launcher.sh python3 -u matmul_tensor_parallel.py
mpiexec -np 8 --ppn 4  ./launcher.sh python3 -u matmul_tensor_parallel.py 
