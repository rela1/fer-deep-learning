import nn
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import tf_model

DATA_DIR = '/media/irelic/Storage/My Documents/Ivan/Škola/FER/9. semestar/Duboko učenje/lab/lab1/data/'
SAVE_DIR = '/media/irelic/Storage/My Documents/Ivan/Škola/FER/9. semestar/Duboko učenje/lab/lab2/results/MNIST_tf_reg_1e-3/'

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['weight_decay'] = 1e-1
config['learning_rate'] = 1e-4

config['conv1_output'] = 16
config['conv1_kernel'] = 5
config['conv1_stride'] = 1
config['pool1_kernel'] = 2
config['pool1_stride'] = 2

config['conv2_output'] = 32
config['conv2_kernel'] = 5
config['conv2_stride'] = 1
config['pool2_kernel'] = 2
config['pool2_stride'] = 2

config['fc_outputs'] = [512]
config['num_classes'] = 10

if __name__ == '__main__':
    np.random.seed(int(time.time() * 1e6) % 2**31)
    dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 1, 28, 28]).transpose((0, 2, 3, 1))
    train_y = dataset.train.labels
    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 1, 28, 28]).transpose((0, 2, 3, 1))
    valid_y = dataset.validation.labels
    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 1, 28, 28]).transpose((0, 2, 3, 1))
    test_y = dataset.test.labels
    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    session = tf.Session()
    inputs = tf.placeholder(dtype=tf.float32, shape=(config['batch_size'], train_x.shape[1], train_x.shape[2], train_x.shape[3]))
    labels = tf.placeholder(dtype=tf.float32, shape=(config['batch_size'], config['num_classes']))
    logits, per_example_loss, loss, weights_collection = tf_model.build_model(inputs, labels, config['num_classes'], config)

    nn.train_tf(train_x, train_y, valid_x, valid_y, session, inputs, labels, logits, loss, weights_collection, config)
    nn.evaluate_tf("Test", test_x, inputs, test_y, labels, session, logits, loss, config)