# Genetic Programming Playground
Performs Neural Architecture Search with a Global Search Space using the NSGA-2 Evolutionary Algorithm (https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf)

# Purpose
The purpose of this project is to find efficient (well-performing but small) neural network architectures using an evolutionary algorithm.
The ultimate intention of this playground project is to analyse the evolutionary process and the resulting best-performing network architectures, 
obtaining insights into the influence and effectiveness of network architectures on model performance.
(Besides this, personal goals included enriching my programming skills in areas such as code consistency & documentation, project design & management)


# Modular representation
ToDo


# Main Details
Convolutional Neural Network architectures (Individuals) are being evaluated on the CIFAR100 dataset.
The multi-objective NSGA-2 algorithm runs with the following two objectives for each CNN model (Individual):
1. Maximize the validation accuracy on the CIFAR100 dataset. Training details below.
2. Minimize the number of trainable parameters used. Hard-capped at 1M.


This version of the playground uses solely Mobile Inverted Residual Bottleneck blocks as the "building bricks"/modules/layers/cells to limit the size of the search space. 
An earlier version had a even more expansive search space with Convolutional, BN, FC, ReLU layers as the smallest "building bricks"/modules/layers/cells but with limited computational resources (1x GTX2060) the evolutionary process was inhibitantly slow.

# Results
ToDo
