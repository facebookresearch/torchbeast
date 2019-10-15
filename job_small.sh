#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=1
#$ -e ./logs/
#$ -o ./logs/
. ./torchbeast_venv/bin/activate
mkdir -p ./logs/
mkdir -p ./runs/
mkdir -p ./output/

python -m torchbeast.monobeast \
     --num_actors 2 \
     --total_steps 30000000 \
     --learning_rate 0.0002 \
     --grad_norm_clipping 1280 \
     --epsilon 0.01 \
     --entropy_cost 0.01 \
     --batch_size 8 \
     --unroll_length 25 \
     --num_threads 4 \
     --xpid "run_small_001" \
     --savedir "./output"
