sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-gt/3s-overlap-split-tf/m123 \
       --postfix _test"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:05:00 --wrap="python3 sft/inference.py \
       --run_name gsm8k-dl/3s-overlap-split-tf/m123 \
       --postfix _test"