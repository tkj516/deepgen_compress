#!/bin/bash

ising_095="/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_{0.95}_200000"
checkpoint_095="dgcspn/ising_0.5_{0.95}_200000/generative/model_2021-05-06_16:19:06.pt"

ising_090="/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_{0.90}_200000"
checkpoint_090="dgcspn/ising_0.5_{0.90}_200000/generative/model_2021-05-06_11:56:42.pt"

ising_085="/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_{0.85}_200000"
checkpoint_085="dgcspn/ising_0.5_{0.85}_200000/generative/model_2021-05-06_19:39:05.pt"

ising_080="/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_{0.80}_200000"
checkpoint_080="dgcspn/ising_0.5_{0.80}_200000/generative/model_2021-05-06_19:39:05.pt"

ising_075="/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_{0.75}_200000"
checkpoint_075="dgcspn/ising_0.5_{0.75}_200000/generative/model_2021-05-06_19:39:05.pt"

ising_070="/fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_{0.70}_200000"
checkpoint_070="dgcspn/ising_0.5_{0.70}_200000/generative/model_2021-05-06_19:39:05.pt"

# Ising

# 0.95
# python demo.py --binary --depthwise --dataset ising --root_dir "$ising_095" --phase test --source_type pgm --stay 0.95 --num_avg $1
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_095" --phase test --source_type spn --checkpoint "$checkpoint_095" --num_avg $1

# 0.90
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_090" --phase test --source_type pgm --stay 0.90 --num_avg $1
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_090" --phase test --source_type spn --checkpoint "$checkpoint_090" --num_avg $1

# 0.85
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_085" --phase test --source_type pgm --stay 0.85 --num_avg $1
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_085" --phase test --source_type spn --checkpoint "$checkpoint_085" --num_avg $1

# 0.80
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_080" --phase test --source_type pgm --stay 0.80 --num_avg $1
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_080" --phase test --source_type spn --checkpoint "$checkpoint_080" --num_avg $1

# 0.75
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_075" --phase test --source_type pgm --stay 0.75 --num_avg $1
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_075" --phase test --source_type spn --checkpoint "$checkpoint_075" --num_avg $1

# 0.70
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_070" --phase test --source_type pgm --stay 0.70 --num_avg $1
python demo.py --binary --depthwise --dataset ising --root_dir "$ising_070" --phase test --source_type spn --checkpoint "$checkpoint_070" --num_avg $1

# MNIST
# python demo.py --binary --depthwise --dataset mnist --root_dir ../../../MNIST --phase test --source_type spn --checkpoint dgcspn/mnist/generative/model_2021-05-06_11:46:44.pt --num_avg $1

