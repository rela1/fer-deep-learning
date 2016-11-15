import nn
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import tf_model
import pickle
import os
import sys

DATA_DIR = '/media/irelic/Storage/My Documents/Ivan/Škola/FER/9. semestar/Duboko učenje/lab/lab2/data/cifar-10-batches-py/'
SAVE_DIR = '/media/irelic/Storage/My Documents/Ivan/Škola/FER/9. semestar/Duboko učenje/lab/lab2/results/MNIST_tf_reg_1e-3_cifar/'

config = {}
config['max_epochs'] = 10
config['batch_size'] = 250
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-7
config['learning_rate'] = {1 : 1e-3, 3 : 1e-4, 5 : 1e-5, 7: 1e-6}

config['conv1_output'] = 16
config['conv1_kernel'] = 5
config['conv1_stride'] = 1
config['pool1_kernel'] = 3
config['pool1_stride'] = 2

config['conv2_output'] = 32
config['conv2_kernel'] = 5
config['conv2_stride'] = 1
config['pool2_kernel'] = 3
config['pool2_stride'] = 2

config['fc_outputs'] = [256, 128]
config['num_classes'] = 10

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict

def one_hot(array, num_classes):
    one_hot_arr = np.zeros((len(array), num_classes))
    one_hot_arr[np.arange(len(array)), array] = 1
    return one_hot_arr

if __name__ == '__main__':
    np.random.seed(int(time.time() * 1e6) % 2**31)
    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)
    test_y_oh = one_hot(test_y, num_classes)

    valid_size = 5000
    train_x, train_y = nn.shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    valid_y_oh = one_hot(valid_y, num_classes)
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    train_y_oh = one_hot(train_y, num_classes)
    data_mean = train_x.mean((0,1,2))
    data_std = train_x.std((0,1,2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    session = tf.Session()
    inputs = tf.placeholder(dtype=tf.float32, shape=(config['batch_size'], train_x.shape[1], train_x.shape[2], train_x.shape[3]))
    labels = tf.placeholder(dtype=tf.float32, shape=(config['batch_size'], config['num_classes']))
    logits, per_example_loss, loss, weights_collection = tf_model.build_model(inputs, labels, config['num_classes'], config)

    nn.train_tf(train_x, train_y_oh, valid_x, valid_y_oh, session, inputs, labels, logits, loss, weights_collection, config)
    nn.evaluate_tf("Test", test_x, inputs, test_y_oh, labels, session, logits, loss, config)
    nn.save_20_misclassified(test_x, inputs, test_y_oh, labels, session, logits, per_example_loss, config, data_mean, data_std)