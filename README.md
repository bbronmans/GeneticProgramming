# Genetic Programming Playground
Performs Neural Architecture Search with a Global Search Space using the [NSGA-2](https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf) Evolutionary Algorithm 

# Purpose
The practical purpose of this project is to find efficient (well-performing but small) neural network architectures using an evolutionary algorithm.
The ultimate intention of this playground project is to analyse the evolutionary process and the resulting best-performing network architectures, 
and thereby obtaining insights into the influence of network architectures on model performance.
(Besides this, personal goals included enriching my programming skills in areas such as code consistency & documentation, project design & management)

# Global search
Most Neural Architecture Searches use cell-based representations to represent neural architectures. Often the search space comprises of several architectural options within a cell, but the number of cells, and connections between them are fixed or limited to a strict, pre-defined set of options. Restricting architectures to have a pre-defined structure drastically decreases the size of the search space, limiting it only to high-potential architectures. In contrast, a global search space does not have any predefined structure. This vastly increases the search space with mainly uncompetitive architectures, but the larger set of options leads to the evaluation of novel, non-traditional architectures which makes the search so much more exciting.

# Modular representation
In this repository, the representation of the neural architectures into a genome is not cell-based but layer-based. Besides the common (Seperable) Convolutional, Fully Connected, ReLu, BatchNorm, Dropout and Pooling layers, there is also a Merge layer that can represent Addition or Multiplication and a [Mobile Inverted Bottleneck](https://arxiv.org/abs/1801.04381) layer (uh-oh, nudging towards Cell-based representation). Mutation options are:

* Mutate the characteristics of a layer (change number of output channels/features or stride, switch between avg/maxpool or add/multiply, etc)
* Add a random new layer
* Remove a layer

# Main Detail
Convolutional Neural Network architectures (Individuals) are being evaluated on the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
The multi-objective NSGA-2 algorithm evaluates the following two fitness criteria for the generated CNN architectures ("Solutions" or "Individuals" in evolutionary terminology) after training for 50 epochs:
1. Error rate on the CIFAR100 dataset (Minimize).
2. Number of trainable parameters in the architecture (Minimize). Hard-capped at 1M.


Note: 
This version of the playground uses mainly Mobile Inverted Residual Bottleneck block and Conv+BN+ReLu layers with other basic layers/"building bricks" commented out in [layer.py](./layer.py#L407) to limit the size of the search space. 
An earlier version had a even more expansive search space with Convolutional, BN, FC, ReLU layers as the smallest layers but with limited computational resources (1x GTX2060) the evolutionary process was inhibitantly slow.

# Results
After training for a few days the original, truly global search space found an architecture that achieved 63.9% validation accuracy with 843K trainable parameters after 50 training epochs. After 17 epochs of the NSGA-2 algorithm (~8 hours on GTX2060), the version of this repositry found an architecture that scores 53.7% validation accuracy with only 297K trainable parameters. For comparison, after 17 NSGA-2 epochs the best performing architecture of the original achieved 43.6%/432K.
