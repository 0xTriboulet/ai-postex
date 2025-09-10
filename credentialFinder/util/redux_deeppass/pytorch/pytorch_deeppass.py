"""
Password Classification Model Training Script

This script trains a binary classifier to detect whether a given word is likely a password.
It performs the following steps:

1. Loads a dataset of words labeled as passwords or non-passwords from a CSV file.
2. Encodes each word as a fixed-length sequence of character indices using a printable character vocabulary.
3. Splits the data into training, validation, and test sets.
4. Constructs a PyTorch `PasswordClassifier` model using an embedding layer and a bidirectional LSTM.
5. Trains the model using weighted binary cross-entropy loss to handle class imbalance.
6. Tracks the best-performing model using validation accuracy.
7. Evaluates the trained model on the test set using accuracy, precision, recall, F1 score, and confusion matrix.
8. Saves the trained model in both `.pt` (PyTorch) and `.onnx` formats.

Constants:
- `PASSWORD_THRESHOLD`: The probability threshold to classify a word as a password.
- `BATCH_SIZE`: Batch size for training.
- `DATASET_PATH`: Path to the input CSV dataset.

Expected CSV Format:
- `word`: The word or token being evaluated.
- `is_password`: A binary label (1 = password, 0 = not password).

Dependencies:
- pandas, numpy, torch, sklearn

Outputs:
- Console metrics during training and testing.
- Saved model: `pytorch_model.pt` and `pytorch_model.onnx`.

Usage:
    python pytorch_deeppass.py

"""
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

import string

PASSWORD_THRESHOLD =  0.5
BATCH_SIZE = 2048
DATASET_PATH = "../dataset.csv"

print(f"[i] Reading data from {DATASET_PATH}")

# load data
df = pd.read_csv(DATASET_PATH, encoding='ansi')

# tokenization
chars = set(''.join(string.printable)) # get the reference chars from printable chars
char2idx = {c: i+2 for i, c in enumerate(sorted(chars))}
char2idx['<PAD>'] = 0
char2idx['<UNK>'] = 1

min_len = 7
max_len = 32

def encode_word(w: str) -> list:
    encoded = [char2idx.get(ch, char2idx['<UNK>']) for ch in w]
    encoded = encoded[:max_len]
    if len(encoded) < max_len:
        encoded += [0]*(max_len - len(encoded)) # pad if our list is not the correct length
    return encoded
    
# split data int X and y; X is all the data our neural network can use to predict y
X = df.word
y = df.is_password.values

# Test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1337, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.01, random_state=1337, shuffle=True)

# Encode after we split to minimize information leakage between test/train
X_train = np.array([encode_word(w) for w in X_train])
X_test = np.array([encode_word(w) for w in X_test])
X_val = np.array([encode_word(w) for w in X_val])

# Validation tensors
X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val).to(torch.float)

# DataSet and DataLoader, a little bit of OOP by convention w/ Pytorch
class PasswordDataset(data.Dataset):
    def __init__(self, X_init, y_init):
        self.X = torch.tensor(X_init, dtype=torch.long) 
        self.y = torch.tensor(y_init, dtype=torch.float) 
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx], self.y[idx]
        
train_dataset = PasswordDataset(X_train, y_train)
test_dataset = PasswordDataset(X_test, y_test)

train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

vocab_size = len(char2idx)
embedding_dim = 20
hidden_dim = 200

# Build our model as an object, by pytorch convention
class PasswordClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate = 0.5):
        super(PasswordClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, 1)        
        self.sigmoid = nn.Sigmoid()
        self.max_len = max_len
        
    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.bilstm(x)
        x = lstm_out[:, self.max_len - 1,:]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
        
model = PasswordClassifier(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 45 # training rounds; worth tinkering with
counter = 0

# Class weights from initial implementation
class_weights = {0: 0.9, 1: 0.1}
criterion = nn.BCELoss(reduction='none')

best_loss = float('inf')
best_acc = 0.0
best_weights = None

# Training
for epoch in range(epochs):
    model.train() # put the model in training mode before we start training
    epoch_accuracy = 0
    print(f"\n\n---[+] Epoch {epoch + 1} / {epochs}---\n")
    total_correct = 0
    total_samples = 0
    
    counter = 0 # TODO: Consider deleting this term, no longer in use
    
    for inputs, targets in train_loader:

        outputs = model(inputs) # get model's garbage prediction
        loss_per_sample = criterion(outputs.squeeze(), targets) # calculate loss based on loss function and outputs vs actual values
        # create a weight tensor for each sample based on its class
        sample_weights = torch.where(targets == 1, torch.tensor(class_weights[1], dtype=torch.float),
                                                   torch.tensor(class_weights[0], dtype=torch.float))

        weighted_loss = (loss_per_sample * sample_weights).mean() # calc weighted loss

        # training accuracy
        preds = (outputs >= PASSWORD_THRESHOLD).float()
        preds = preds.view(-1) # change the view to match axis of targets
        correct = (torch.eq(preds, targets)).sum().item()
      
        batch_acc = correct / outputs.numel() # calculate batch accuracy
        batch_loss = weighted_loss.item()
        
        # iff loss decreased and acc increased, update model weights, otherwise iterate again
        if batch_acc > epoch_accuracy:
            epoch_accuracy = batch_acc
            print(f"[i] Epoch {epoch + 1} / {epochs}, Counter: {counter}, Batch Loss: {batch_loss: .4f}, Batch Accuracy: {batch_acc:.4f}, Best Epoch Accuracy: {epoch_accuracy:.4f}")

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        total_correct += correct
        total_samples += (outputs == outputs).sum().item()
        
        counter += 1
        
    
    # eval model based on validation accuracy
    model.eval()
    
    y_pred = model(X_val)
    y_pred = (y_pred >= PASSWORD_THRESHOLD).float()
    y_pred = y_pred.view(-1)
    
    y_data_val = y_val
    acc = (torch.eq(y_pred, y_data_val)).sum().item() / y_data_val.numel()
    
    print(f"Validation accuracy: {acc}")
    
    if acc > best_acc:
        best_acc  = acc
        best_weights = copy.deepcopy(model.state_dict())
    
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # epoch_accuracy = total_correct / total_samples
    print(f">>> [i] Epoch {epoch + 1} / {epochs}, Best Loss: {best_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.4f}, Best Accuracy: {best_acc:.4f} <<<\n\n")
     
model.eval() # put the model in eval mode before we eval

all_preds = []
all_targets = []

# Testing
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        preds = (outputs.squeeze() >= PASSWORD_THRESHOLD).long().numpy() # if > 50% chance this is a password
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
  
cm = confusion_matrix(all_targets, all_preds)
acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds)
f1 = f1_score(all_targets, all_preds)
rec = recall_score(all_targets, all_preds)

print(f"Confusion matrix:\n{cm}")
print(f"Accuracy score:\n{acc}")
print(f"Precision score:\n{prec}")
print(f"F1 score:\n{f1}")
print(f"Recall score:\n{rec}")

# save as pytorch model
torch.save(model.state_dict(), "pytorch_model.pt")

print(f"\n[i] Saved model pytorch_model.pt")

# dummy input
dummy_input = torch.randint(low=2, high=vocab_size, size=(1, max_len), dtype=torch.long)

# save as onnx
torch.onnx.export(
    model,
    dummy_input,
    "pytorch_model.onnx",
    input_names = ["input"],
    output_names = ["output"],
    opset_version= 8,
    dynamo=False,
    do_constant_folding=True,
    export_params=True,
    )
    
print(f"[i] Saved model pytorch_model.onnx")

