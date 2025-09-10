# Deeppass Password Classifier
=====================================

This script trains and evaluates a deep neural network to classify strings as passwords or not. This model architecture ports the design of SpectreOps' BiLSTM model `https://posts.specterops.io/deeppass-finding-passwords-with-deep-learning-4d31c534cd00`

## Requirements
------------

* See `requirements.txt`

## Usage
-----

### Training

To train the model, simply run `python pytorch_deeppass.py` in your terminal. The script will train the model on the password dataset and evaluate its performance on a validation set.

### Saving the Model

The script saves the trained model as a PyTorch model (`pytorch_model.pt`) and an ONNX model (`pytorch_model.onnx`).

## Notes
----

* The script uses a threshold of 50% to classify passwords in training.
* The script assumes that the password dataset is stored in a directory called `../datasetv.csv`.
* The script evaluates the model's performance using accuracy, precision, recall, and F1 score.
