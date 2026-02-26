import yaml
import argparse
import subprocess
import pandas as pd
import os
from evaluate_metrics import evaluate_metrics

def run_inference(model_type, model_path, h5_path, ds_name):
    cmd = [
        "python", "run_inference.py",
        "--model_type", model_type,
        "--model_path", model_path,
        "--h5_path", h5_path,
        "--ds_name", ds_name
    ]
    print(f"\n Running inference: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_evaluation(csv_path):
    df = pd.read_csv(csv_path)
    return evaluate_metrics(df)

def main():
    parser = argparse.ArgumentParser(description="Run batched inference using multiple models on multiple datasets")
    parser.add_argument('--dataset_type', type=str, required=True, choices=['neutral_aimnet2_supported', 
        'neutral_others', 'charged_aimnet2_supported', 'charged_uma_supported'], 
            help = 'Dataset type to use: neutral_aimnet2_supported, neutral_others, charged_aimnet2_supported or charged_uma_supported')
    args = parser.parse_args()

    if args.dataset_type == 'neutral_aimnet2_supported':
        config_file = "config_neutral_aimnet2_supported.yaml"
        final_df_name = "metrics_summary_neutral_aimnet2_supported.csv"
    elif args.dataset_type == 'neutral_others':
        config_file = "config_neutral_others.yaml"
        final_df_name = "metrics_summary_neutral_others.csv"
    elif args.dataset_type == 'charged_aimnet2_supported':
        config_file = "config_charged_aimnet2_supported.yaml"
        final_df_name = "metrics_summary_charged_aimnet2_supported.csv"
    elif args.dataset_type == 'charged_uma_supported':
        config_file = "config_charged_uma_supported.yaml"
        final_df_name = "metrics_summary_charged_uma_supported.csv"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    results = []

    for dataset in config["datasets"]:
        name = dataset["name"]
        h5_path = dataset["h5_path"]

        for model in config["models"]:
            model_type = model["type"]
            model_path = model["path"]
            model_name = os.path.splitext(os.path.basename(model_path))[0]

            # run inference
            run_inference(model_type, model_path, h5_path, name)
            csv_file = f"{model_name.upper()}_Inference_{name}_intE.csv"
            csv_path = os.path.join("outputs", csv_file)
    
            if not os.path.exists(csv_path):
                print(f" Missing output CSV: {csv_path}, skipping evaluation.")
                continue
    
            # run evaluation
            metrics = run_evaluation(csv_path)
            metrics["Dataset"] = name
            metrics["ModelType"] = model_type
            metrics["ModelName"] = model_name
            results.append(metrics)

    # save all results to a pandas dataframe
    final_df = pd.DataFrame(results)
    final_df = final_df[["Dataset", "ModelType", "ModelName", "R2", "Pearson_R2", "RMSE (kcal/mol)", "MAE (kcal/mol)"]]
    final_df.to_csv(final_df_name, index=False)

if __name__ == "__main__":
    main()
