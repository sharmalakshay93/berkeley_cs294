#!/usr/bin/env bash

declare -a experts=("Ant-v2" "HalfCheetah-v2" "Hopper-v2" "Humanoid-v2" "Reacher-v2" "Walker2d-v2")

for expert in "${experts[@]}"
do
	echo "training bc_model for $expert"
	python3 bc.py --expert ${expert} --only_final_results True
done