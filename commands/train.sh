#!/bin/bash  
    
hf_model_name="google/gemma-2b-it"
model_folder="gemma-2b-it"  

experiment_name="step_4"
steps=("steps_4" "steps_34" "steps_78" "steps_234" "steps_678" "steps_5678" "steps_45678") # "steps_2345678")
runs=(1 2 3 4 5)
for run in "${runs[@]}"; do  
    for step in "${steps[@]}"; do  
        sbatch --gpus=rtx_4090:1 --mem-per-cpu=8G --wrap="python3 sft/train.py --data_path /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step}/${step}.json --run_name /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step} --model_name_or_path ${hf_model_name}"
    done  
done 

experiment_name="step_23"  
steps=("steps_2" "steps_23" "steps_234" "steps_2345" "steps_23456" "steps_234567")  
runs=(1 2 3 4 5)
for run in "${runs[@]}"; do  
    for step in "${steps[@]}"; do  
        sbatch --gpus=rtx_4090:1 --mem-per-cpu=8G --wrap="python3 sft/train.py --data_path /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step}/${step}.json --run_name /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step} --model_name_or_path ${hf_model_name}"
    done  
done  

model_folder="phi-3-mini-instruct"  
hf_model_name="microsoft/Phi-3-mini-128k-instruct"  

experiment_name="step_4"
steps=("steps_4" "steps_34" "steps_78" "steps_234" "steps_678" "steps_5678" "steps_45678") # "steps_2345678")
runs=(1 2 3 4 5)
for run in "${runs[@]}"; do  
    for step in "${steps[@]}"; do  
        sbatch --gpus=rtx_4090:1 --mem-per-cpu=8G --wrap="python3 sft/train.py --data_path /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step}/${step}.json --run_name /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step} --model_name_or_path ${hf_model_name}"
    done  
done 

experiment_name="step_23"  
steps=("steps_2" "steps_23" "steps_234" "steps_2345" "steps_23456" "steps_234567")  
runs=(1 2 3 4 5)
for run in "${runs[@]}"; do  
    for step in "${steps[@]}"; do  
        sbatch --gpus=rtx_4090:1 --mem-per-cpu=8G --wrap="python3 sft/train.py --data_path /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step}/${step}.json --run_name /cluster/work/lawecon/Work/mbraasch/output/${model_folder}/${experiment_name}/${run}/${step} --model_name_or_path ${hf_model_name}"
    done  
done