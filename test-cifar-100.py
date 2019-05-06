import pickle
import numpy as np
import random
from collections import Counter
# python 3
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_list = []
filename_list = []
batch_label_list = []
data_list = []
test_data = unpickle('test_batch')
train_data = {}
for i in range(1, 6):
    tmp = unpickle('data_batch_%d'%i)
    print(type(tmp[b'labels']))
    label_list.extend(tmp[b'labels'])
    filename_list.extend(tmp[b'filenames'])
    batch_label_list.extend(tmp[b'batch_label'])
    data_list.extend(tmp[b'data'])

train_data['labels'] = label_list
train_data['filenames'] = filename_list
train_data['batch_label'] = batch_label_list
train_data['data'] = data_list


for key in train_data:
    print(key)

print(train_data['labels'])
print(train_data['batch_label'])
z_index = [i for i, x in enumerate(train_data['labels']) if x == 0]
print(z_index)
random.shuffle(z_index)
print(z_index)
print(len(z_index))


counter = Counter(train_data['labels'])
print(counter)