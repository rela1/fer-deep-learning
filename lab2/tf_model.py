import tensorflow.contrib.layers as layers
import tensorflow as tf

def build_model(inputs, labels, num_classes, config):
    with tf.contrib.framework.arg_scope([layers.convolution2d],
        padding='SAME', activation_fn=tf.nn.relu, 
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(config['weight_decay'])):
            net = layers.convolution2d(inputs=inputs, num_outputs=config['conv1_output'], kernel_size=config['conv1_kernel'], stride=config['conv1_stride'], variables_collections=['weights'], scope='conv1')
            net = layers.max_pool2d(inputs=net, kernel_size=config['pool1_kernel'], stride=config['pool1_stride'], scope='pool1')
            net = layers.convolution2d(inputs=net, num_outputs=config['conv2_output'], kernel_size=config['conv2_kernel'], stride=config['conv2_stride'], variables_collections=['weights'], scope='conv2')
            net = layers.max_pool2d(inputs=net, kernel_size=config['pool2_kernel'], stride=config['pool2_stride'], scope='pool2')

    with tf.contrib.framework.arg_scope([layers.fully_connected], activation_fn=tf.nn.relu,
        weights_initializer=layers.variance_scaling_initializer(), 
        weights_regularizer=layers.l2_regularizer(config['weight_decay'])):
        net = layers.flatten(net)
        net = layers.fully_connected(net, config['fc1_output'], scope='fc1', variables_collections=['weights'])

    logits = layers.fully_connected(net, config['num_classes'], activation_fn=None, scope='logits', variables_collections=['weights'])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
    return logits, loss, tf.get_collection('weights')