# Descriptor Analysis for Domain Generalization 

This repository contains code for understanding descriptors in domain generalization algorithms quantitatively and qualitatively. The baseline algorithms and datasets have been been adapted from [DomainBed](https://github.com/facebookresearch/DomainBed).

Currently, 2 algorithms have been included:
* Empirical Risk Minimization (ERM, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
* Interdomain Mixup (Mixup, [Yan et al., 2020](https://arxiv.org/abs/2001.00677))

The dataset being used currently is:

* DomainNet ([Peng et al., 2019](http://ai.bu.edu/M3SDA/))

- Checkout `commands.sh` on how to train and generate descriptors. 
- Requirements for pip can be found inside `requirements.txt`

For the above algorithms and datasets, the descriptors can be found in the `extracted` folder.

This work has been done as part of the CS 444 Deep Learning for Computer Vision project.