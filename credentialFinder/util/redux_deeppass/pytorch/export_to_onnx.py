'''Stand-alone helpper script to convert .pt model to specific opset version of .onnx format.'''
import torch
import torch.nn as nn

import string

# tokenization
chars = set(''.join(string.printable)) # get the reference chars from printable chars
char2idx = {c: i+2 for i, c in enumerate(sorted(chars))}
char2idx['<PAD>'] = 0
char2idx['<UNK>'] = 1

min_len = 7
max_len = 32

vocab_size = len(char2idx)
embedding_dim = 20
hidden_dim = 200

#model architecture
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

# dummy input
dummy_input = torch.randint(low=2, high=vocab_size, size=(1, max_len), dtype=torch.long)

model = PasswordClassifier(vocab_size, embedding_dim, hidden_dim)

# Load the PyTorch model from the saved file
model.load_state_dict(torch.load("pytorch_model.pt"))

model.eval()

print("Loaded model")

model_path = "deepseek_model.onnx"

# Export the ONNX model to a new file with OpSet 8
model_onnx = torch.onnx.export(
    model,
    dummy_input,
    model_path,
    input_names = ["input"],
    output_names = ["output"],
    opset_version= 9,
    dynamo=False,
    do_constant_folding=True,
    export_params=True,
    )

print(f"Saved ONNX model: {model_path}")