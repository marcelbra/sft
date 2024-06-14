sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.05-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.1-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.15-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.2-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.25-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.3-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.05-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.1-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.15-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.2-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.25-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m234/m234-tf-0.3-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.05-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.1-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.15-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.2-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.25-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.3-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.05-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.1-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.15-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.2-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.25-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m234/m234-tf-0.3-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m123/s1_next_step_predictions_s1_test.json \
       --postfix _test"
