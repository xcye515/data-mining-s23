# data-mining-s23
Source code for my research project: ES Optimization for the MNIST Classification Task

Course: IEOR E4540 Data Mining

Instructor: Prof. Krzysztof Choromanski

Author: Xingchen Ye

# Abstract: 
This project investigates the performance of Evolution Strategies (ES) optimization in comparison to backpropagation for training neural networks on the MNIST classification task. Three studies are conducted to evaluate the impact of various factors on the performance of ES-trained models. The first study compares ES optimization with backpropagation using Stochastic Gradient Descent (SGD) and a fixed learning rate. We find that ES-trained models exhibit better convergence, while backpropagation with TensorFlow built-in optimizer achieves faster training time. The second study explores the effect of different numbers of perturbations (N = 10, 50, 100) on ES training, revealing that higher perturbation numbers result in faster and better convergence at the cost of increased training time. The third study compares vanilla ES, antithetic pairs, and FD-style gradient sensing methods, showing that with reward normalization, the three Monte Carlo estimators yield similar convergence curves and downstream accuracies. We further experiment with antithetic rewards using σR scaling, which leads to slower convergence but higher validation accuracy. 
