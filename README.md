# Molecular Interaction Energy Inference

This repository provides a unified interface to run **inference** using **AIMNet2** or **MACE-OFF** models on molecular dimer systems stored in HDF5 format.

No model training or development is included — this repo is strictly for **inference using pre-trained models**.

---

## 📁 Repository Structure

```
.
├── models/                # Pre-trained AIMNet2/MACE-OFF models
├── datasets/              # Input datasets in HDF5 format
├── outputs/               # Inference result csv files will be saved here
├── aimnet2_inference.py   # AIMNet2 inference pipeline
├── maceoff_inference.py   # MACE-OFF inference pipeline
├── run_inference.py       # Unified command-line entry point
├── evaluate_metrics.py    # Script to evaluate predicted vs reference interaction energies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── requirements.txt       # Python dependencies
```

---

## 🚀 Usage

### Run inference:
```bash
python run_inference.py \
  --model_type {aimnet2 or maceoff} \
  --model_path models/{your desired model} \
  --h5_path datasets/sample_dataset.h5 \
  --ds_name sample_dataset
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
