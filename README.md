NEURAL NETWORK PROJECT 

This project implements a simple 2-layer neural network from scratch to classify handwritten numbers (MNIST dataset)
- Input layer -> 784 neurons (28x28 flattened)
- Hidden layer -> 10 neurons ReLU activated
- Input layer -> 10 neurons Softmax activated
Training uses gradient descent with back-propagation
Network is implemented using NumPy and Pandas - no ML libraries

Setup and dependencies: 
Using venv: 
python -m venv venv 
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

The dataset is not included (too big) but was downloaded via: https://www.kaggle.com/c/digit-recognizer

How to run: 
- download mnist dataset and place path to file in pd.read_csv() function
- run the script: python main.py
- script will load and preprocess data, train neural network for set of iterations and print accuracy every 10th iteration
