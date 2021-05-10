#!/bin/bash

python demo.py --binary --depthwise --dataset mnist --root_dir ../../../MNIST --phase test --source_type spn --checkpoint dgcspn/mnist/generative/model_2021-05-06_11:46:44.pt --num_avg 100

