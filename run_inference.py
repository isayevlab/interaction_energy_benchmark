import argparse
import os
from aimnet2_inference import AIMNET2_Inference
from maceoff_inference import MACEOFF_Inference
from maceomol_inference import MACEOMOL_Inference

def main():
    parser = argparse.ArgumentParser(description="Run inference using AIMNet2, MACE-OFF or MACE-OMOL models")
    parser.add_argument('--model_type', type=str, required=True, choices=['aimnet2', 'maceoff', 'maceomol'],
                        help='Model type to use: aimnet2, maceoff or maceomol')
    parser.add_argument('--model_path', type=str, default=None,
                        help='File path for the intended model')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='File path to the input HDF5 dataset file')
    parser.add_argument('--ds_name', type=str, required=True,
                        help='Dataset name')
    args = parser.parse_args()

    if args.model_type == 'aimnet2':
        if not args.model_path or not os.path.isfile(args.model_path):
            raise ValueError("A valid --model_path must be provided for AIMNet2 model")
        print(f"Running AIMNet2 on dataset: {args.ds_name}")
        model = AIMNET2_Inference(args.model_path, args.h5_path, args.ds_name)
        results = model.run_inference()
        model.save_results(results)

    elif args.model_type == 'maceoff':
        if not args.model_path or not os.path.isfile(args.model_path):
            raise ValueError("A valid --model_path must be provided for MACE-OFF model")
        print(f"Running MACE-OFF on dataset: {args.ds_name}")
        model = MACEOFF_Inference(args.model_path, args.h5_path, args.ds_name)
        results = model.run_inference()
        model.save_results(results)

    elif args.model_type == 'maceomol':
        if not args.model_path or not os.path.isfile(args.model_path):
            raise ValueError("A valid --model_path must be provided for MACE-OMOL model")
        print(f"Running MACE-OMOL on dataset: {args.ds_name}")
        model = MACEOMOL_Inference(args.model_path, args.h5_path, args.ds_name)
        results = model.run_inference()
        model.save_results(results)


if __name__ == "__main__":
    main()
