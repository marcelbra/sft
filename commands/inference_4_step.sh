#!/bin/bash  

experiment_name="step_4"
steps=("steps_4567") # "steps_456" "steps_4567" "steps_45678")
runs=(1) # 2 3 4 5)
for run in "${runs[@]}"; do  
    for step in "${steps[@]}"; do  
        # sbatch ---gpus=rtx_4090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py --run_name /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/${experiment_name}/${run}/${step} --model_name_or_path google/gemma-2b-it --postfix _test_4_steps --data_dir /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/step_4/4_steps_test.json"
        # sbatch ---gpus=rtx_4090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py --run_name /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/${experiment_name}/${run}/${step} --model_name_or_path microsoft/Phi-3-mini-128k-instruct --postfix _test_4_steps --data_dir /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/step_4/4_steps_test.json"
        # sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py --run_name /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/${experiment_name}/${run}/${step} --model_name_or_path google/gemma-2b-it --postfix _test_4_steps --data_dir /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/step_4/4_steps_test.json"
        sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py --run_name /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/${experiment_name}/${run}/${step} --model_name_or_path microsoft/Phi-3-mini-128k-instruct --postfix _test_4_steps --data_dir /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/step_4/4_steps_test.json"
    done  
done 