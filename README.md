# GNN_combo Repository

This repository contains all the necessary scripts for training and preprocessing the data for a Graph Neural Network (GNN) model. It is designed to handle and process health-related data, specifically from the MIMIC-III dataset, and utilizes BERT for additional preprocessing steps.

## Repository Structure

Below is a brief overview of the main components of the repository:


GNN_combo/
├── model.py
├── train_combo.py
├── utils.py
├── data/
├── get_cls/
│   └── get_cls.py
├── mimic_GNN/
│   ├── post_data/
│   ├── dx_map.p
│   ├── lab_map.p
│   ├── proc_map.p
│   ├── test.tfrecord
│   ├── test_csr.pkl
│   ├── train.tfrecord
│   ├── train_csr.pkl
│   ├── validation.tfrecord
│   └── validation_csr.pkl
├── preprocess_mimic.py
├── preprocess_BERT/
│   ├── mk_X_BERT.py
│   ├── mk_X_BERT_matched.py
│   └── preprocess_NOTEEVENTS.py
└── train.py
└── train_BERT.py
└── train_latest.py


## Files and Directories

- `model.py`: This file contains the main structure of the Graph Neural Network (GNN) model.
- `train_combo.py`: This script is used for training the GNN model.
- `utils.py`: This file contains utility functions used throughout the repository.
- `data/`: This directory should contain your input data.
- `get_cls/`: This directory contains `get_cls.py`, a script for obtaining the classes from the data.
- `mimic_GNN/`: This directory contains files related to the MIMIC-III dataset, including preprocessed data, mappings, and TFRecord files.
- `preprocess_mimic.py`: This script preprocesses the MIMIC-III dataset for use in the GNN model.
- `preprocess_BERT/`: This directory contains scripts (`mk_X_BERT.py`, `mk_X_BERT_matched.py`, `preprocess_NOTEEVENTS.py`) for preprocessing data with BERT.
- `train.py`: This is an alternative script for training the GNN model.
- `train_BERT.py`: This script is used for training the BERT model.
- `train_latest.py`: This script is presumably the latest or most updated version of the training script.

## Usage

First, make sure your raw data is located in the `data/` directory. The scripts are designed to read from this location. Then, follow the sequence of steps below:

1. Run `preprocess_mimic.py` to preprocess your MIMIC-III data.
2. Run the scripts in `preprocess_BERT/` to further preprocess your data with BERT.
3. Run `get_cls.py` to obtain the classes from your data.
4. Depending on your requirements, run either `train.py`, `train_combo.py`, `train_BERT.py`, or `train_latest.py` to train your GNN model.

## Requirements

Please ensure you have the necessary dependencies installed. If not, install them with:
'''
pip install -r requirements.txt
'''
