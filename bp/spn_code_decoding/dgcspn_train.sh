#!/bin/bash

python dgcspn_train.py --depthwise --binary --dataset ising --root_dir /fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_\{0.55\}_200000 --epochs 100 --gpu_id 1
python dgcspn_train.py --depthwise --binary --dataset ising --root_dir /fs/data/tejasj/Masters_Thesis/deepgen_compress/bp/ising_0.5_\{0.50\}_200000 --epochs 100 --gpu_id 1
