wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar -xzf hpl-2.3.tar.gz
cd hpl-2.3
./configure --prefix=$PWD/build CC=mpicc
make -j 8 install