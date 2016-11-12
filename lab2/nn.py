import os
import math
import tensorflow as tf
import numpy as np
import skimage as ski
import skimage.io
import tf_model


def forward_pass(net, inputs):
  output = inputs
  for layer in net:
    output = layer.forward(output)
  return output


def backward_pass(net, loss, x, y):
  grads = []
  grad_out = loss.backward_inputs(x, y)
  if loss.has_params:
    grads += loss.backward_params()
  for layer in reversed(net):
    grad_inputs = layer.backward_inputs(grad_out)
    if layer.has_params:
      grads += [layer.backward_params(grad_out)]
    grad_out = grad_inputs
  return grads

def sgd_update_params(grads, config):
  lr = config['lr']
  for layer_grads in grads:
    for i in range(len(layer_grads) - 1):
      params = layer_grads[i][0]
      grads = layer_grads[i][1]
      #print(layer_grads[-1], " -> ", grads.sum())
      params -= lr * grads
      

def draw_conv_filters(epoch, step, layer, save_dir):
  C = layer.C
  w = layer.weights.copy()
  num_filters = w.shape[0]
  k = int(np.sqrt(w.shape[1] / C))
  w = w.reshape(num_filters, C, k, k)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  #for i in range(C):
  for i in range(1):
    img = np.zeros([height, width])
    for j in range(num_filters):
      r = int(j / cols) * (k + border)
      c = int(j % cols) * (k + border)
      img[r:r+k,c:c+k] = w[j,i]
    filename = '%s_epoch_%02d_step_%06d_input_%03d.png' % (layer.name, epoch, step, i)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def train(train_x, train_y, valid_x, valid_y, net, loss, config):
  lr_policy = config['lr_policy']
  batch_size = config['batch_size']
  max_epochs = config['max_epochs']
  save_dir = config['save_dir']
  num_examples = train_x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  for epoch in range(1, max_epochs+1):
    if epoch in lr_policy:
      solver_config = lr_policy[epoch]
    cnt_correct = 0
    #for i in range(num_batches):
    # shuffle the data at the beggining of each epoch
    permutation_idx = np.random.permutation(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]
    #for i in range(100):
    for i in range(num_batches):
      # store mini-batch to ndarray
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
      logits = forward_pass(net, batch_x)
      loss_val = loss.forward(logits, batch_y)
      # compute classification accuracy
      yp = np.argmax(logits, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()
      grads = backward_pass(net, loss, logits, batch_y)
      sgd_update_params(grads, solver_config)

      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))
      if i % 100 == 0:
        draw_conv_filters(epoch, i*batch_size, net[0], save_dir)
        #draw_conv_filters(epoch, i*batch_size, net[3])
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    evaluate("Validation", valid_x, valid_y, net, loss, config)
  return net


def evaluate(name, x, y, net, loss, config):
  print("\nRunning evaluation: ", name)
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  cnt_correct = 0
  loss_avg = 0
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits = forward_pass(net, batch_x)
    yp = np.argmax(logits, 1)
    yt = np.argmax(batch_y, 1)
    cnt_correct += (yp == yt).sum()
    loss_val = loss.forward(logits, batch_y)
    loss_avg += loss_val
    #print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg)

def train_tf(train_x, train_y, valid_x, valid_y, session, inputs, labels, logits, loss, weights_collection, config):
  batch_size = config['batch_size']
  max_epochs = config['max_epochs']
  save_dir = config['save_dir']
  num_examples = train_x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
  train_op = optimizer.minimize(loss)
  session.run(tf.initialize_all_variables())
  for epoch in range(1, max_epochs+1):
    cnt_correct = 0
    # shuffle the data at the beggining of each epoch
    permutation_idx = np.random.permutation(num_examples)
    train_x = train_x[permutation_idx]
    train_y = train_y[permutation_idx]
    #for i in range(100):
    for i in range(num_batches):
      # store mini-batch to ndarray
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
      logits_val, loss_val, conv1_weights_val, _ = session.run([logits, loss, weights_collection[0], train_op], feed_dict={inputs:batch_x, labels:batch_y})
      # compute classification accuracy
      yp = np.argmax(logits_val, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()
      if i % 5 == 0:
        print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, np.average(loss_val)))
      if i % 100 == 0:
        draw_conv_filters_tf(epoch, i*batch_size, conv1_weights_val, save_dir)
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    evaluate_tf("Validation", valid_x, inputs, valid_y, labels, session, logits, loss, config)


def evaluate_tf(name, x, inputs, y, labels, session, logits, loss, config):
  print("\nRunning evaluation: ", name)
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  cnt_correct = 0
  loss_avg = 0
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits_val, loss_val = session.run([logits, loss], feed_dict={inputs:batch_x, labels:batch_y})
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    cnt_correct += (yp == yt).sum()
    loss_avg += np.average(loss_val)
  valid_acc = cnt_correct / num_examples * 100
  loss_avg /= num_batches
  print(name + " accuracy = %.2f" % valid_acc)
  print(name + " avg loss = %.2f\n" % loss_avg)

def draw_conv_filters_tf(epoch, step, weights, save_dir):
  w = weights.copy()
  num_filters = w.shape[3]
  num_channels = w.shape[2]
  k = w.shape[0]
  assert w.shape[0] == w.shape[1]
  w = w.reshape(k, k, num_channels, num_filters)
  w -= w.min()
  w /= w.max()
  border = 1
  cols = 8
  rows = math.ceil(num_filters / cols)
  width = cols * k + (cols-1) * border
  height = rows * k + (rows-1) * border
  img = np.zeros([height, width, num_channels])
  for i in range(num_filters):
    r = int(i / cols) * (k + border)
    c = int(i % cols) * (k + border)
    img[r:r+k,c:c+k,:] = w[:,:,:,i]
  filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
  if num_channels == 1:
    img = np.reshape(img, (height, width))
  ski.io.imsave(os.path.join(save_dir, filename), img)