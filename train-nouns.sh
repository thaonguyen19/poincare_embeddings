#!/bin/sh

# Get number of threads from environment or set to default
if [ -z "$NTHREADS" ]; then
   NTHREADS=5
fi

echo "Using $NTHREADS threads"

# make sure OpenMP doesn't interfere with pytorch.multiprocessing
export OMP_NUM_THREADS=1

# The hyperparameter settings reproduce the mean rank results 
# reported in [Nickel, Kiela, 2017]
# For MAP results, replace the learning rate parameter with -lr 2.0

python3 embed.py \
       -dim 10 \
       -lr 1.0 \
       -epochs 1500 \
       -negs 50 \
       -burnin 20 \
       -nproc "${NTHREADS}" \
       -distfn poincare \
       -dset package/functions_04182018_train.tsv \
       -valset package/functions_04182018_val \
       -dupset package/functions_04182018_duplicate_train \
       -fout packages_latest.pth \
       -batchsize 100 \
       -eval_each 25 \
