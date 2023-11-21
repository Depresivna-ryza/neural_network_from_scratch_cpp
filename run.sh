#!/bin/bash

# module add gcc-10.2


echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
g++ -Wall -std=c++20 -O3 -fopenmp src/main.cpp src/miscellaneous.hpp src/neuralnetwork.hpp src/matrix.hpp -o network

echo "#################"
echo "     RUNNING     "
echo "#################"

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
./network

echo "#################"
echo "      DONE       "
echo "#################"

echo "Test data:"
python3 evaluator/evaluate.py data/test_predictions.csv data/fashion_mnist_test_labels.csv
echo "Train data:"
python3 evaluator/evaluate.py data/train_predictions.csv data/fashion_mnist_train_labels.csv


