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




### Loading PACS Descriptors

```
def extract_data(data, merge_groups=True, transform=None, ):
    zs, ys, preds, gs, logits = data['feature'], data['label'], data['pred'], data['group'], data['logits']
    if transform is not None:
        zs = transform(zs)
    #     gs = gs % 2
    return zs, ys, gs, preds, logits
    
PATH_TO_FILE_TRAIN = "put path to .p train file here"
PATH_TO_FILE_TEST = "put path to .p test file here"

train_data = pickle.load(open(PATH_TO_FILE_TRAIN, 'rb'))
test_data = pickle.load(open(PATH_TO_FILE_TEST', 'rb'))

train_gs = train_data['group']
n_train = len(train_gs)
groups, counts = np.unique(train_data['group'], return_counts=True, axis=0)
n_groups = len(groups)
n_classes = len(np.unique(train_data['label']))
  
  
# Load train and test data according to extract_data function
  
zs, ys, gs, preds, logits = extract_data(train_data)

test_zs, test_ys, test_gs, test_preds, test_logits = extract_data(test_data)
  
 ```
