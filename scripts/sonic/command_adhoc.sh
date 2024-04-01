#! /bin/bash

python experiments.py --model=resnet --model_type=resnet18  --dataset=cifar10 \
    --alg=adhocSL \
    --lr=0.001 \
    --batch-size=64 \
    --epochs=10 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-#label1 \
    --beta=0.5\
    --device='cuda' \
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=2 \
    --init_seed=4 \
    --cut_a=1 \
    --cut_b=9 \
    --warmup=0 \
    --sl_step=1
