
## Neural Network From Scratch
  - Implementation of a neural network from scratch
    (no high-level libraries).
  - Architecture is a Multi-Layer Perceptron classifier
  - On the given dataset, the MLP achieves 90% accuracy on the test set and
    trains within 10 minutes

### DATASET
Fashion MNIST (https://arxiv.org/pdf/1708.07747.pdf) a modern version of a
well-known MNIST (http://yann.lecun.com/exdb/mnist/). It is a dataset of
Zalando's article images ‒ consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes. The dataset is in CSV format. There
are four data files included:  
 - `fashion_mnist_train_vectors.csv`   - training input vectors
 - `fashion_mnist_test_vectors.csv`    - testing input vectors
 - `fashion_mnist_train_labels.csv`    - training labels
 - `fashion_mnist_test_labels.csv`     - testing labels

### FEATURES
  - MLP forward and backpropagation using matrix multiplication
  - Classifier softmax head
  - Hyperparameter search
  - Tests

### PREREQUISITES

Make sure to have the following installed:

`g++` with support for -std=c++20

`python3` version 3.10+

### USAGE

To run the training and evaluator:
```
 ./bash.sh
```
