import pickle
import argparse

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from utils import extract_pacs_data
import numpy as np
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--groups', type=str, required=False, help='Comma-separated group ids')
args = parser.parse_args()

data = pickle.load(open(args.data, 'rb'))  
  
# Load train and test data according to extract_data function
features, labels, groups, preds, logits = extract_pacs_data(data)
try:
    if args.groups is not None:
        group_filter = [int(gid) for gid in args.groups.split(',')]
        indices = []
        for gid in args.groups.split(','):
            gid = int(gid)
            # print(np.where(groups == gid)[0].shape)
            indices += np.where(groups == gid)[0].tolist()
        indices = (np.array(indices),)
        print(indices[0].shape)
        features, labels, groups, preds, logits = features[indices], labels[indices], groups[indices], preds[indices], logits[indices]
except:
    # print(traceback.format_exc())
    print('groups arg not in the correct format. should be csv of group ids.')

cf_matrix = confusion_matrix(y_true=labels, y_pred=preds)
unique_labels = np.unique(labels).tolist()
sorted_labels = sorted(unique_labels, key=lambda l: cf_matrix[l, l], reverse=True)
disp = ConfusionMatrixDisplay.from_predictions(y_true=labels, y_pred=preds, labels=sorted_labels, cmap='Blues', include_values=False)
plt.title(args.data)
plt.show()