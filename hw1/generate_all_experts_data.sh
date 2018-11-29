#!/usr/bin/env bash

declare -a experts=("Ant-v2" "HalfCheetah-v2" "Hopper-v2" "Humanoid-v2" "Reacher-v2" "Walker2d-v2")

for expert in "${experts[@]}"
do
	echo "generating data for expert $expert"
	python3 run_expert.py experts/${expert}.pkl $expert --num_rollouts 20
done