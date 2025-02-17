#!/bin/bash
#partition method can be "dir".
python main_mypfl.py --model 'resnet18' \
--dataset 'officehome' \
--partition_method 'homo' \
--partition_alpha '0.3' \
--batch_size 128 \
--lr 0.1 \
--lr_decay 0.998 \
--epochs 5 \
--client_num_in_total 100 --frac 0.1 \
--comm_round 500 \
--dense_ratio 0.5 \
--anneal_factor 0.5 \
--seed 2022 \
--cs 'random' \
--dis_gradient_check \
--different_initial
