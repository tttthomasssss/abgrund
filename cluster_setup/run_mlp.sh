#!/bin/bash

# Parameters for Sun Grid Engine submission
# ============================================

# Shell to use
#$ -S /bin/bash

# All paths relative to current working directory
#$ -cwd

# List of queues
#$ -q nlp-amd,parallel.q,inf.q,serial.q

# Define parallel environment for N cores
#$ -pe openmp 1

# Send mail to. (Comma separated list)
#$ -M thk22@sussex.ac.uk

# When: [b]eginning, [e]nd, [a]borted and reschedules, [s]uspended, [n]one
#$ -m n

# Validation level (e = reject on all problems)
#$ -w e

# Merge stdout and stderr streams: yes/no
#$ -j yes

# do not alter the task IDs below
#$ -t 1
#$ -tc 1
python /mnt/lustre/scratch/inf/thk22/code/neuralnets-sentimentanalysis/mlp/MLP.py $SGE_TASK_ID
