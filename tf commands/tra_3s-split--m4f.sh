sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-added/m4-final-tf-0.05-added.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-added/m4-final-tf-0.1-added.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-added/m4-final-tf-0.15-added.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-added/m4-final-tf-0.2-added.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-added/m4-final-tf-0.25-added.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-added/m4-final-tf-0.3-added.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-added";

sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-replaced/m4-final-tf-0.05-replaced.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-replaced/m4-final-tf-0.1-replaced.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-replaced/m4-final-tf-0.15-replaced.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-replaced/m4-final-tf-0.2-replaced.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-replaced/m4-final-tf-0.25-replaced.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-replaced/m4-final-tf-0.3-replaced.json \
       --run_name gsm8k-gt/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-replaced";
       
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-added/m4-final-tf-0.05-added.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-added/m4-final-tf-0.1-added.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-added/m4-final-tf-0.15-added.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-added/m4-final-tf-0.2-added.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-added/m4-final-tf-0.25-added.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-added";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-added/m4-final-tf-0.3-added.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-added";

sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-replaced/m4-final-tf-0.05-replaced.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.05-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-replaced/m4-final-tf-0.1-replaced.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.1-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-replaced/m4-final-tf-0.15-replaced.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.15-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-replaced/m4-final-tf-0.2-replaced.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.2-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-replaced/m4-final-tf-0.25-replaced.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.25-replaced";
sbatch --gpus=rtx_3090:1  --mem-per-cpu=8G \
       --wrap="python3 sft/train_target.py \
       --data_path /cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-replaced/m4-final-tf-0.3-replaced.json \
       --run_name gsm8k-dl/3s-overlap-split-tf/m4-final/m4-final-tf-0.3-replaced";
