# Molecular Interaction Energy Inference

This repository provides a unified interface to run **inference** using **AIMNet2**, **MACE-OFF**, **MACE-OMOL** or **UMA-OMOL** models on molecular dimer systems stored in HDF5 format.

No model training or development is included — this repo is strictly for **inference using pre-trained models**.

---

## 📁 Repository Structure

```
.
├── models/                			# Pre-trained AIMNet2/MACE-OFF/MACE-OMOL/UMA-OMOL models
├── outputs/               			# Inference result csv files will be saved here
├── datasets.tar.gz        			# Input datasets in HDF5 format (compressed format)
├── aimnet2_inference.py   			# AIMNet2 inference pipeline
├── maceoff_inference.py   			# MACE-OFF inference pipeline
├── maceomol_inference.py  			# MACE-OMOL inference pipeline
├── umaomol_inference.py  			# UMA-OMOL inference pipeline
├── run_inference.py       			# Unified command-line to run inference
├── batched_inference.py   			# Inference script for multiple datasets at once (via configuration file)
├── config_charged_aimnet2_supported.yaml    	# Configuration yaml file for charged datasets (AIMNet2), model type and path, etc.
├── config_charged_uma_supported.yaml    	# Configuration yaml file for charged datasets (UMA), model type and path, etc.
├── config_neutral_aimnet2_supported.yaml    	# Configuration yaml file for neutral datasets (AIMNet2), model type and path, etc.
├── config_neutral_others.yaml    		# Configuration yaml file for neutral datasets (Others), model type and path, etc.
├── evaluate_metrics.py    			# Script to evaluate predicted vs reference interaction energies
├── README.md              			# This file
├── .gitignore             			# Git ignore rules
└── requirements.txt       			# Python dependencies
```

---

## 🚀 Usage

### Run inference for a single dataset:
```bash
python run_inference.py \
  --model_type {aimnet2 or maceoff or maceomol or umaomol} \
  --model_path models/{your desired model} \
  --h5_path datasets/sample_dataset.h5 \
  --ds_name sample_dataset
```

### Run inference for multiple datasets at once:
```bash
python batched_inference.py --dataset_type {charged_aimnet2_supported or charged_uma_supported or neutral_aimnet2_supported or neutral_others}
```

### Evaluate results:
```bash
python evaluate_metrics.py \
  --csv_path outputs/{result csv file}
```

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
