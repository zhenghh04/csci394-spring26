#!/bin/bash
#
# Run HPL on a laptop (single node) with 8 processes.
# Memory size is 64GB. 
HPL_BIN=${HPL_BIN:-"../hpl-2.3/build/bin/xhpl"}
mpiexec -np 8 "$HPL_BIN"