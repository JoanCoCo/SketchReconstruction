import os
import time
import json
import subprocess
import argparse

# Program to used for running the training under a given configuration. 
# If the training breaks, this program will automatically relaunch it.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="guardian.py")
    parser.add_argument('--config', type=str, default=None)
    ARGS = parser.parse_args()
    if ARGS.config is not None:
        data = json.load(open(ARGS.config, 'r'))
        for key in data:
            ARGS.__dict__[key] = data[key]
    else:
        print("GUARDIAN: No configurarion file was provided, aborting.")
        exit()

    print("GUARDIAN: Launching training.")
    start_time = time.time()
    arguments = ["python", "train.py"] + ARGS.arguments

    cont = True
    while cont:
        cont = False
        try:
            subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError:
            cont = True
            print("GUARDIAN: Training crash detected, launching again.")
    end_time = time.time()
    print("GUARDIAN: Training finished after {:.2f} min, closing.".format((end_time - start_time)/60.0))
    exit()