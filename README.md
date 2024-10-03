# Intent-Classification-with-FNN

# Intent Classification with Feedforward Neural Network

## Project Overview

This project implements a **Feedforward Neural Network (FFNN)** for intent classification using a provided dataset. The network is trained with backpropagation to classify user intents from sentences. The dataset consists of sentences and their associated intent categories. The task is to classify the sentences into different intent categories using basic neural network principles implemented from scratch with **NumPy**.

## Dataset

The dataset contains sentences and their corresponding intents. Example entries are:
- **Input**: "what movie times are at bow tie cinemas"
  - **Intent**: `SearchScreeningEvent`
- **Input**: "can you put this xandee"
  - **Intent**: `AddToPlaylist`

The data is preprocessed into a bag-of-words format for model training.

## Objectives

1. **Bag-of-Words Representation**: Convert text data into a bag-of-words matrix and intent labels into one-hot encoded vectors.
2. **Activation Functions**: Implement activation functions like **ReLU** and **Softmax**, along with their derivatives, and visualize them.
3. **Feedforward Neural Network**:
   - Implement forward propagation for a 3-layer neural network with 150 neurons in the hidden layer.
   - Use **ReLU** as the activation function in the hidden layer and **Softmax** for the output layer.
4. **Backpropagation**:
   - Implement backpropagation to compute the gradients of the cost function and update the network weights using batch gradient descent.
   - Track the cost function over epochs and use it to evaluate the model's learning progress.
5. **Training and Evaluation**: 
   - Train the model for 1000 epochs and evaluate the model’s accuracy.
   - Optionally implement mini-batch and stochastic gradient descent as bonus tasks.

## Project Structure

```
.
├── data/
│   ├── dataset.csv               # The dataset file containing sentences and intents
├── model/
│   ├── __init__.py
│   ├── ffnn.py                   # Feedforward Neural Network implementation
│   ├── model_utils.py            # Utility functions (activation functions, bag-of-words)
├── assignment2.py                # Main script to run the FFNN model
├── utils.py                      # Utility functions for data preprocessing
├── helper.py                     # Helper functions for training, evaluation, etc.
```

## Instructions

### 1. Preprocess Data
- The text data is converted into a bag-of-words representation using `bag_of_words_matrix()` from `model_utils.py`.
- A preprocessing step is applied where words occurring less than 2 times are replaced with an `<UNK>` token.

### 2. Implement Neural Network
- The neural network is defined in `ffnn.py` and consists of:
  - An input layer (bag-of-words)
  - A hidden layer with **ReLU** activation
  - An output layer with **Softmax** activation
- Initialize the model parameters with random values and perform forward propagation using the `forward()` function.

### 3. Train the Model
- Train the model using **batch gradient descent** with backpropagation. The training process computes gradients for the weights and updates them over 1000 epochs.
- Use the `--train` flag to train the model:
   ```bash
   python assignment2.py --train
   ```
  
### 4. Evaluate the Model
- After training, evaluate the accuracy of the model using the `predict()` function.
- Use the `--test` flag to evaluate the trained model:
   ```bash
   python assignment2.py --test
   ```

### 5. Bonus Tasks
- Implement **mini-batch** and **stochastic gradient descent** for further experimentation with learning rates and batch sizes:
   ```bash
   python assignment2.py --minibatch
   ```

## Results and Discussion

- **Performance**: The model is evaluated based on its accuracy on the dataset. Since the same dataset is used for both training and testing, the results may not generalize well.
- **Loss Analysis**: The cost function is plotted across epochs to monitor the learning process.

## Requirements

To run this project, install the following dependencies:
```bash
pip install numpy matplotlib
```
