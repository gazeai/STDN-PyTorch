# STDN-PyTorch
This is an unofficial implementation of "On Disentangling Spoof Traces for Generic Face Anti-Spoofing" in PyTorch

[[Paper](http://cvlab.cse.msu.edu/pdfs/liu_stehouwer_liu_eccv2020.pdf)]
[[Official Implementation in TensorFlow](https://github.com/yaojieliu/ECCV20-STDN)]

## Requirements
- python 3.6+
- PyTorch 1.6.0
- easydict
- TensorFlow 2+

## Data Preparation
- Create keypoints for both Live and Spoof faces
- Create a csv with Column Names being 'rgb_path', 'keypoints', 'label'
- For labels, Spoof is 1 and Real is 0
- Equally Sample the dataset so that there is equal number of spoof and equal number of real.

# Configurations
To change your configurations, open `config.py` and for setting up data paths, change in `flags.data_config`. You can change other parameters too based on your need.

## How to train?
Once data preparation is done, and data path is created, run the following code:

```python train.py```

All your checkpoints will be saved under the directory `ckpts`. However, you can change the path from `config.py`. the file names will be in the following format:

```model_epoch-number_val-accuracy_val-apcer_val-bpcer.pth```

## How to test?
Still in progress.

## Help taken from
1. On Disentangling Spoof Traces for Generic Face Anti-Spoofing [[1]](#1).
2. [[Official Implementation in TensorFlow](https://github.com/yaojieliu/ECCV20-STDN)]

## Citations
<a id="1">[1]</a> 
Yaojie Liu, Joel Stehouwer, Xiaoming Liu. 

    
   
