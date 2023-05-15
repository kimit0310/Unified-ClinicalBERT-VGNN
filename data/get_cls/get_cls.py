import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import threading
import subprocess
import time
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_nvidia_smi():
    while True:
        subprocess.run(['nvidia-smi'])
        time.sleep(600)  # Sleep for 10 minutes


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data.to_dict('records')
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        text = record['TEXT']
        hadm_id = record['HADM_ID']

        if not isinstance(text, str) or not text.strip():
            return None
        inputs = self.tokenizer(text, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.squeeze(0).to('cuda') for k, v in inputs.items()}
        inputs["hadm_id"] = hadm_id

        return inputs


def extract_features_and_save(dataset, model, tokenizer, output_file):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    model = model.to('cuda')
    model.eval()
    results = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to('cuda')
                      for k, v in batch.items() if k != "hadm_id"}
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for idx, hadm_id in enumerate(batch["hadm_id"]):
                result = {"HADM_ID": hadm_id}
                result.update({f"feature_{i}": value for i,
                              value in enumerate(features[idx])})
                results.append(result)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


print("Starting nvidia-smi monitoring...")
threading.Thread(target=run_nvidia_smi, daemon=True).start()

print("Loading tokenizer and model...")
checkpoint_path = '/scratch/ka2705/NLP/NYU_NLU_BERT_GNN_COMBO/data/preprocess_BERT/checkpoint-1024'
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained(checkpoint_path)
model = model.to('cuda')

datasets = {
    "train": "train_matched.csv",
    "val": "val_matched.csv",
    "test": "test_matched.csv",
}

for dataset_name, dataset_file in datasets.items():
    print(f"Loading {dataset_name} dataset...")
    data = pd.read_csv(dataset_file)
    dataset = CustomDataset(data, tokenizer)
    print(f"Extracting features for {dataset_name} dataset...")
    extract_features_and_save(
        dataset, model, tokenizer, f"{dataset_name}_features.csv")

print("Feature extraction complete.")
