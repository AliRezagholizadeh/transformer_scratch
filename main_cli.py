
import argparse
from train_singleN_multiG import singleN_multipleGPU_main_interface
from train_singleG import train_single_gpu_interface
import sys

def main_cli():
    parser = argparse.ArgumentParser(description="Training AI model.")
    parser.add_argument("--multi-gpu", action='store_true', help="Bool. Whether to use multi gpu")
    parser.add_argument("--multi-node", action='store_true', help="Bool. Whether to use multi gpu")
    parser.add_argument("--config-path", "-c", help="Specify an output file")

    args = parser.parse_args()

    # default config file or inserted one
    config_path = "config.yml"
    if args.config_path:
        config_path = args.config_path


    if args.multi_gpu:
        print(f"multi_gpu: {args.multi_gpu}")
        if args.multi_node:
            print("Multi Node, multi GPU selected.")
            pass
        else:  # when single-node selected
            print("Single Node, multi GPU selected.")

            # call the relevant function from train_singleN_multiG script
            singleN_multipleGPU_main_interface(config_path)

    else:
        train_single_gpu_interface(config_path)


    if args.config_path:
        print(f"config_path: {args.config_path}")



if __name__ == "__main__":
    main_cli()