import yaml
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
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    results = []

    for dataset in config["datasets"]:
        name = dataset["name"]
        model_type = dataset["model_type"]
        model_path = dataset["model_path"]
        h5_path = dataset["h5_path"]

        # run inference
        run_inference(model_type, model_path, h5_path, name)
        csv_file = f"{model_type.upper()}_Inference_{name}_intE.csv"
        csv_path = os.path.join("outputs", csv_file)

        if not os.path.exists(csv_path):
            print(f" Missing output CSV: {csv_path}, skipping evaluation.")
            continue

        # run evaluation
        metrics = run_evaluation(csv_path)
        metrics["Dataset"] = name
        metrics["ModelType"] = model_type
        metrics["ModelName"] = os.path.basename(model_path)
        results.append(metrics)

    # save all results to a pandas dataframe
    final_df = pd.DataFrame(results)
    final_df = final_df[["Dataset", "ModelType", "ModelName", "R2", "RMSE (kcal/mol)", "MAE (kcal/mol)"]]
    final_df.to_csv("metrics_summary.csv", index=False)

if __name__ == "__main__":
    main()
