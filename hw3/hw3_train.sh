#!/bin/bash
cat ./split_w/x* >> ./hw3_model_weights.h5
CUDA_VISIBLE_DEVICES=0	python3 hw3_train.py $1 
