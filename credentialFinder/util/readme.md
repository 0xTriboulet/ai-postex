**Redux DeepPass - A PyTorch implementation of SpectreOps' DeepPass Bidirectional LSTM (BiLSTM) Model**
==========================
Subfolders have relevant `readme.md`s, but the core capability is built in `.\redux_deeppass\pytorch`.

**Project Background**
---------------------

This Python script is based on the SpectreOps' DeepPass project, which was initially developed by [GhostPack](https://github.com/GhostPack). This repository provides an implementation of a deep learning model using TensorFlow and
Keras to predict text sequences. It's an extension of the original [DeepPass](https://github.com/GhostPack/DeepPass) project, with the primary benefit being usage outside of a Docker.

**Project Modifications**
---------------------

- Extracted model from dockerized implementation
- Ported model architecture from Keras to Pytorch
- Implemented data generation step
- Implemented model conversion from `.pt` to `.onnx`
