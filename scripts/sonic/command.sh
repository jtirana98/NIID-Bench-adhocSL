#! /bin/bash
echo 'seed-5-fedprox-cifar-c=1'

python experiments.py --model=simple-cnn --dataset=cifar10 \
    --alg=adhocSL \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=10 \
    --n_parties=10 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-#label1  \
    --beta=0.5\
    --device='cpu'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=1 \
    --init_seed=4\
    --cut_a=2 \
    --cut_b=3 \
    --warmup=1 \
    --sl_step=1
