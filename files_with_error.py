"""
ChatGPT Prompt: Write a Python script for me. You should be able to set a path and the program should go into that directory and find all files that have a structure like this "slurm-<some number>.out" for example a file could be "slurm-3717004.out". For each of the files that match that load it (it's just a text file). Go through that file and check if the string "torch.cuda.OutOfMemoryError" is in the file. For all files where that is the case, print to the console. This program should be able to be called from command line with "python3 find_files_with_error --path /here/the/path".
"""

import os
import argparse

def find_files_with_error(path):
    # Loop through the files in the specified directory
    for filename in os.listdir(path):
        # Check if the filename matches the pattern slurm-<number>.out
        if filename.startswith('slurm-') and filename.endswith('.out'):
            file_path = os.path.join(path, filename)
            # Open and read the file
            with open(file_path, 'r') as file:
                contents = file.read()
                # Check for the specific error string
                if "torch.cuda.OutOfMemoryError" in contents:
                    print(file_path)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Find files with specific error.')
    parser.add_argument('--path', type=str, required=True, help='Path to the directory to search in.')
    args = parser.parse_args()
    find_files_with_error(args.path)

"""
sbatch --gpus=rtx_4090:1 --wrap="python3 sft/files_with_error.py --path /cluster/home/mbraasch"
"""