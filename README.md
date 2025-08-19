# SFNet: A Lightweight Method for River Ice Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight deep learning model for segmenting river ice from remote sensing imagery. This repository contains the official code for training, testing, and running inference with SFNet.


## Features

* **Lightweight Architecture**
* **High Accuracy**

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/SFNet.git](https://github.com/your-username/SFNet.git)
    cd SFNet
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

The dataset used for this project is publicly available.

This repository includes several utility scripts to help prepare the data.

* `caculate_weights.py`: Calculates class weights to handle class imbalance in the dataset.
* `convertlabel.py`: Converts annotation files to the required format for training.
* `label_process.py`: A script for adjusting labels.

## Usage

### Training

To train the model from scratch, you can run the `train.py` script. Key hyperparameters such as learning rate, batch size, and number of epochs can be configured directly within the script.

```bash
python train.py
```
### Testing

if you use --best, it will test the best model checkpoint found within the last 200 epochs from your specified model path.

```bash
python test.py
```

### Predicting

```bash
python predict.py
```
