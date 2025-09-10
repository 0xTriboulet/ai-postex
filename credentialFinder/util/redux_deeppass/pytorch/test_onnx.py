''' Testing basic ONNX functionality with opencv, can also be used to validate the output probability against other (C++) implementations of model usage'''

import onnx
import cv2 as cv # opencv
import numpy as np
import string 

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
    
text = ["checksum"] # ["morning"] # example text

# turn our list into a proper array    
input_array = np.array([encode_word(w) for w in text], dtype=np.float32).reshape(1, max_len)

print(input_array)

model_path = "deepseek_model.onnx"

print("[i] Loading model: ")
# load the model
model = cv.dnn.readNetFromONNX(model_path)

# Set the model's input
model.setInput(input_array)

print(f"[i] Inference using model: {model_path}")

# Get model prediction
output = model.forward()

print(output)
