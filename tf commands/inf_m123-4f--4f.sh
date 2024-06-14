sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.05-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.1-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.15-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.2-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.25-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.3-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.05-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.1-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.15-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.2-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.25-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/m123-m4f-tf/m4-final/m4-final-tf-0.3-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.05-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.1-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.15-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.2-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.25-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.3-added \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.05-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.1-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.15-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.2-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.25-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/m123-m4f-tf/m4-final/m4-final-tf-0.3-replaced \
       --previous_next_steps /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/m123-m4f-tf/m123/next_step_predictions_test.json \
       --postfix _test"
