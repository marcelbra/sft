#!/bin/bash  
    
experiment_name="step_4"
steps=("steps_45" "steps_456" "steps_4567" "steps_45678")
runs=(1 2 3 4 5)

# hf_model_name="google/gemma-2b-it"
# model_folder="gemma-2b-it"  
# for run in "${runs[@]}"; do  
#     for step in "${steps[@]}"; do  
#         sbatch --gpus=1 --gres=gpumem:24g --mem-per-cpu=8G --wrap="python3 sft/train.py --data_path /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step}/${step}.json --run_name /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step} --model_name_or_path ${hf_model_name}"
#     done  
# done 

model_folder="phi-3-mini-instruct"  
hf_model_name="microsoft/Phi-3-mini-128k-instruct"
for run in "${runs[@]}"; do  
    for step in "${steps[@]}"; do  
        sbatch --gpus=1 --gres=gpumem:24g --mem-per-cpu=8G --wrap="python3 sft/train.py --data_path /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step}/${step}.json --run_name /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step} --model_name_or_path ${hf_model_name}"
    done  
done