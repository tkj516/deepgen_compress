#!/bin/bash

for i in {0..0.95..0.05};
  do 
     python gen_ising_dataset.py --p 0.5 --stay $i --num_images 200000 --root ising_0.5_$i_200000
& done
