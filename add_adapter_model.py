import os
import glob

def add_adapter_model_folder_to(target_dir):

    # Recursively find all directories that have "checkpoint" in their name in the target directory and get their full name including the part with the "checkpoint" in it
    checkpoint_dirs = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(target_dir)
        for name in dirs
        if "checkpoint" in name
    ]

    for checkpoint_dir in checkpoint_dirs:
        adapter_model_dir = os.path.join(checkpoint_dir, "adapter_model")
        if os.path.exists(adapter_model_dir):
            continue
        os.makedirs(adapter_model_dir)
        for file in glob.glob(os.path.join(checkpoint_dir, "*")):
            # Do not rename the adapter_model directory itself
            if file.endswith("adapter_model"):
                continue
            os.rename(file, os.path.join(adapter_model_dir, os.path.basename(file)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", help="The directory that contains the checkpoints", default="/cluster/work/lawecon/Work/mbraasch/output")
    args = parser.parse_args()
    add_adapter_model_folder_to(args.target_dir)

"""
sbatch --wrap="python3 sft/add_adapter_model.py"
"""