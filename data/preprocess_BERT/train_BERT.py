import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import subprocess
import time
import threading
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score
from torch.nn import CrossEntropyLoss

def run_nvidia_smi():
    while True:
        subprocess.run(['nvidia-smi'])
        time.sleep(120)  # Sleep for 2 minutes (120 seconds)

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
        label = record['LABEL']
        
        if not isinstance(text, str) or not text.strip():
            return None
        
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        
        return inputs

print("Starting nvidia-smi monitoring...")
threading.Thread(target=run_nvidia_smi, daemon=True).start()

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)

print("Loading datasets...")
train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')

print("Creating CustomDataset instances...")
train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)

print("Creating Trainer and TrainingArguments...")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_steps=512,
    save_steps=512,
    eval_steps=512,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="auprc",
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()

    # Calculate the probabilities for the positive class
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    positive_probabilities = probabilities[:, 1]

    # Calculate the AUPRC
    auprc = average_precision_score(labels, positive_probabilities)

    return {"accuracy": accuracy, "auprc": auprc}

def data_collator(batch):
    batch = [item for item in batch if item is not None]
    return tokenizer.pad(batch, return_tensors="pt")

# Calculate class weights (adjust)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor([0.0165, 0.9835]).to(device)

# Create a custom Trainer class
class CustomTrainer(Trainer):
    def __init__(self, *args, loss_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_function(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Instantiate the weighted cross-entropy loss function
loss_function = CrossEntropyLoss(weight=class_weights)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    loss_function=loss_function,
)
print("Starting training...")
trainer.train()
print("Training complete.")