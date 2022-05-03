import pickle
import argparse
from utils import extract_pacs_data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
args = parser.parse_args()

data = pickle.load(open(args.data, 'rb'))  
  
# Load train and test data according to extract_data function
features, labels, groups, preds, logits = extract_pacs_data(data)
n_samples = features.shape[0]
group_ids, group_counts = np.unique(groups, return_counts=True, axis=0)
avg_acc = 0
worst_group_acc = 99
worst_group_id = 99
print('Data file: {}'.format(args.data))
for gid, gcount in zip(group_ids, group_counts):
    indices = np.where(groups == gid)
    acc = np.sum(np.equal(labels[indices], preds[indices]))/gcount
    if acc < worst_group_acc:
        worst_group_acc = acc
        worst_group_id = gid
    avg_acc += acc
    print('Group {}: count={}, accuracy={}'.format(gid, gcount, acc))
print('Worst group: id={}, accuracy={}'.format(worst_group_id, worst_group_acc))
avg_acc /= len(group_ids)
print('Average accuracy = {}'.format(avg_acc))