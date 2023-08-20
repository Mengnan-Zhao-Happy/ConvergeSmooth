# ConvergeSmooth
Code for [Fast Adversarial Training with Smooth Convergence]
## Introduction
> 	Fast adversarial training (FAT) is beneficial for improving the adversarial robustness of neural networks.
	However, previous FAT work suffers from catastrophic overfitting at large perturbation budgets, \ie the adversarial robustness of models declines to near zero during training. 
	To address this, we analyze the training process of prior FAT work and observe that catastrophic overfitting is accompanied by the appearance of loss convergence outliers.
	Therefore, we argue a moderately smooth loss convergence process will be a stable FAT process that solves catastrophic overfitting.
    To obtain a smooth loss convergence process, we propose a novel oscillatory constraint (dubbed ConvergeSmooth) to limit the loss difference between adjacent epochs. The convergence stride of ConvergeSmooth is introduced to balance convergence and smoothing.
Likewise, we design weight centralization without introducing additional hyperparameters other than the loss balance coefficient.
	Our proposed methods are attack-agnostic and thus can improve the training stability of various FAT techniques.
## Requirements
Python </br>
Pytorch </br>
## Train
> waiting....

## Test
> python test.py






