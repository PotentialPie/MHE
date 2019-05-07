import pickle
import numpy as np
import random
from collections import Counter
# python 3
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_single_imbalance(train_data, single_class, single_num):
    z_index = [i for i, x in enumerate(train_data['labels']) if x == single_class]
    other_index = [i for i, x in enumerate(train_data['labels']) if x != single_class]
    print(len(z_index))
    print(len(other_index))
    # print(z_index)
    random.shuffle(z_index)
    # print(z_index)
    other_index.extend(z_index[:single_num])
    print(len(other_index))

    random.shuffle(other_index)
    other_index = np.array(other_index)
    print(other_index.shape)
    print(train_data['labels'].shape)
    print(train_data['data'].shape)
    # print(other_index)
    train_data['labels'] = train_data['labels'][other_index]
    # train_data['filenames'] = train_data['filenames'][other_index]
    # train_data['batch_label'] = train_data['batch_label'][other_index]
    train_data['data'] = train_data['data'][other_index]
    print(len(train_data['labels']))
    print(len(train_data['data']))
    return train_data

def get_multiple_imbalance(train_data, num_step=500):
    all_left_index = []
    for class_index in range(10):
        z_index = [i for i, x in enumerate(train_data['labels']) if x == class_index]
        print(len(z_index))
        random.shuffle(z_index)
        print(z_index)
        all_left_index.extend(z_index[:num_step*(class_index+1)])
        print(len(all_left_index))
    print(all_left_index)
    random.shuffle(all_left_index)
    print(all_left_index)
    all_left_index = np.array(all_left_index)
    train_data['labels'] = train_data['labels'][all_left_index]
    # train_data['filenames'] = train_data['filenames'][all_left_index]
    # train_data['batch_label'] = train_data['batch_label'][all_left_index]
    train_data['data'] = train_data['data'][all_left_index]
    print(len(train_data['labels']))
    print(len(train_data['data']))
    return train_data

def read_all_train_data():
    label_list = []
    filename_list = []
    batch_label_list = []
    data_list = []
    train_data = {}
    for i in range(1, 6):
        tmp = unpickle('data_batch_%d' % i)
        print(type(tmp[b'labels']))
        label_list.extend(tmp[b'labels'])
        filename_list.extend(tmp[b'filenames'])
        batch_label_list.extend(tmp[b'batch_label'])
        data_list.extend(tmp[b'data'])

    train_data['labels'] = np.array(label_list)
    train_data['filenames'] = np.array(filename_list)
    train_data['batch_label'] = np.array(batch_label_list)
    train_data['data'] = np.array(data_list)

    for key in train_data:
        print(key)

    # print(train_data['labels'])
    # print(train_data['batch_label'])
    counter = Counter(train_data['labels'])
    print(counter)
    print(type(train_data['labels']))
    return train_data

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 10])
b = np.array([3, 5, 7])
print(a[b])

test_data = unpickle('test_batch')

train_data = read_all_train_data()
single_imbalance_train_data = get_single_imbalance(train_data, 0, 500)
print(len(single_imbalance_train_data['labels']))

print(single_imbalance_train_data['labels'][0])
print(single_imbalance_train_data['data'][0])
print(single_imbalance_train_data['labels'].shape[0])


# train_data = read_all_train_data()
# multiple_imbalance_train_data = get_multiple_imbalance(train_data, num_step=500)
# print(len(multiple_imbalance_train_data['labels']))







