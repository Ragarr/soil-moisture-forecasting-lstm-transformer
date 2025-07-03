# Deep Learning for Robust Soil Moisture Forecasting Using LSTM and Transformer

**Authors:** Raúl Aguilar, Miguel A. Patricio, et al.  
**Conference:** SOCO 2025

---

This repository contains the source code and instructions to reproduce the experiments from the paper:

> **Deep Learning for Robust Soil Moisture Forecasting Using LSTM and Transformer**  
>
> Raúl Aguilar, Miguel A. Patricio, José M. Molina, Antonio Berlanga, Sergio Zubelzu  
>
> Accepted at SOCO 2025.

## Overview

This project aims to predict soil moisture time series using LSTM and Transformer models, focusing on noisy real-world sensor data from the Duero basin (Spain).  
It includes robust data preprocessing, model training, hyperparameter optimization (Optuna), and evaluation scripts.

### Workflow diagram

![diagram-export-3-7-2025-10_59_40](https://github.com/user-attachments/assets/ab834b9d-e1f9-4fd6-a4f9-b0734d1722db)

---

## Repository Structure

```
.
├── data # Data and scripts for dataset creation and preprocessing
│   ├── generate_dataset.py
│   ├── humidity
│   │   └── raw.csv
│   ├── meteo
│   │   ├── abr_meteo24.csv
│   │   └── ...
│   └── segmentate_data.py
├── LICENSE
├── README.md
├── requirements.txt
└── src 
    ├── models.py  # Models definition and utilities
    ├── graph_best_trial_bloque.py  # Visualization of best trial results
    └── search_train_hiperparameters.py # Main script for hyperparameter optimization and training
```


---

## Getting Started

### 1. Clone this repository

```bash
git clone https://github.com/Ragarr/soil-moisture-forecasting-lstm-transformer
cd soil-moisture-forecasting-lstm-transformer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data

Run the data preprocessing script to generate the dataset:
```bash
python data/generate_dataset.py
```
Then segment the data to extract useful subsets:
```bash
python data/segmentate_data.py
```

### 4. Training and Hyperparameter Optimization
Run the main script to start hyperparameter optimization and model training:
```bash
python src/search_train_hiperparameters.py --model_type transformer --horizon 8
```
Available options:

* --model_type: lstm, transformer, or dummy

* --horizon: Number of steps to predict (e.g., 1, 8, 24)

* --output_dir: (optional) Where to save results/checkpoints/logs

Checkpoints, metrics, and logs will be created under experiment-* directories.

### 5. Evaluation and Results
- The script will output best metrics and save the best model parameters and metrics to disk.

- The script graph_best_trial_bloque.py can be used to visualize the best trial results. And output the best model's predictions and metrics.

## Citation
If you use this code or base your research on it, please cite:

#TODO: Add citation information here
