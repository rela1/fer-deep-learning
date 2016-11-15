from sklearn.metrics import confusion_matrix
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import math
import tensorflow as tf
import numpy as np
import skimage as ski
import skimage.io
import tf_model

def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

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
    train_x, train_y = shuffle_data(train_x, train_y)
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
  learning_rate = tf.placeholder(tf.float32)
  learning_rate_map = config['learning_rate']
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss)
  session.run(tf.initialize_all_variables())
  plot_data = {}
  plot_data['train_loss'] = []
  plot_data['valid_loss'] = []
  plot_data['train_acc'] = []
  plot_data['valid_acc'] = []
  plot_data['lr'] = []
  for epoch in range(1, max_epochs+1):
    if epoch in learning_rate_map:
      learning_rate_value = learning_rate_map[epoch]
    cnt_correct = 0
    train_x, train_y = shuffle_data(train_x, train_y)
    for i in range(num_batches):
      batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
      batch_y = train_y[i*batch_size:(i+1)*batch_size, :]
      start_time = time.time()
      logits_val, loss_val, conv1_weights_val, _ = session.run([logits, loss, weights_collection[0], train_op], feed_dict={inputs:batch_x, labels:batch_y, learning_rate:learning_rate_value})
      duration = time.time() - start_time
      yp = np.argmax(logits_val, 1)
      yt = np.argmax(batch_y, 1)
      cnt_correct += (yp == yt).sum()
      if i % 5 == 0:
        sec_per_batch = float(duration)
        print("epoch %d, step %d/%d, batch loss = %.2f (%.3f sec/batch)" % (epoch, i*batch_size, num_examples, loss_val, sec_per_batch))
      if i % 100 == 0:
        draw_conv_filters_tf(epoch, i*batch_size, conv1_weights_val, save_dir)
      if i > 0 and i % 50 == 0:
        print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
    print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
    train_loss, train_acc = evaluate_tf("Train", train_x, inputs, train_y, labels, session, logits, loss, config)     
    valid_loss, valid_acc = evaluate_tf("Validation", valid_x, inputs, valid_y, labels, session, logits, loss, config)   
    plot_data['train_loss'] += [train_loss]
    plot_data['valid_loss'] += [valid_loss]
    plot_data['train_acc'] += [train_acc]
    plot_data['valid_acc'] += [valid_acc]
    plot_data['lr'] += [session.run(optimizer._lr_t)]
  plot_training_progress(config['save_dir'], plot_data)
  draw_conv_filters_tf(-1, -1, conv1_weights_val, save_dir)


def evaluate_tf(name, x, inputs, y, labels, session, logits, loss, config):
  print("\nRunning evaluation: ", name)
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  number_of_classes = config['num_classes']
  conf_matrix = np.zeros((number_of_classes, number_of_classes))
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  loss_avg = 0
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits_val, loss_val = session.run([logits, loss], feed_dict={inputs:batch_x, labels:batch_y})
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    conf_matrix_batch = confusion_matrix(yt, yp, labels=np.arange(10))
    np.add(conf_matrix, conf_matrix_batch, conf_matrix)
    loss_avg += loss_val
  loss_avg /= num_batches
  total_conf_matrix_sum = np.sum(conf_matrix)
  row_conf_matrix_sum = np.sum(conf_matrix, axis = 1)
  column_conf_matrix_sum = np.sum(conf_matrix, axis = 0)
  diagonal_conf_matrix_sum = np.sum(np.diag(conf_matrix))
  acc = diagonal_conf_matrix_sum / total_conf_matrix_sum
  prec = [conf_matrix[i][i] / column_conf_matrix_sum[i] for i in range(number_of_classes)]
  rec = [conf_matrix[i][i] / row_conf_matrix_sum[i] for i in range(number_of_classes)]
  print(name + " accuracy = %.2f" % acc)
  print(name + " per class precision = %s" % prec)
  print(name + " per class recall = %s" % rec)
  print(name + " avg loss = %.2f\n" % loss_avg)
  return loss_avg, acc


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

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.pdf')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)

def save_20_misclassified(x, inputs, y, labels, session, logits, loss, config, img_mean, img_std):
  batch_size = config['batch_size']
  num_examples = x.shape[0]
  num_channels = x.shape[3]
  width = x.shape[2]
  height = x.shape[1]
  save_dir = config['save_dir']
  print("\nSaving 20 misclassified images in: ", save_dir)
  assert num_examples % batch_size == 0
  num_batches = num_examples // batch_size
  losses_with_info = []
  for i in range(num_batches):
    batch_x = x[i*batch_size:(i+1)*batch_size, :]
    batch_y = y[i*batch_size:(i+1)*batch_size, :]
    logits_val, loss_val = session.run([logits, loss], feed_dict={inputs:batch_x, labels:batch_y})
    yp = np.argmax(logits_val, 1)
    yt = np.argmax(batch_y, 1)
    losses_with_info += [(loss_val[j], i * batch_size + j, yt[j], np.argsort(logits_val[j])[-3:]) for j in range(batch_size) if yp[j] != yt[j]]
  losses_with_info.sort(key=itemgetter(0))
  for loss_with_info in losses_with_info[-20:]:
    image = x[loss_with_info[1]]
    image *= img_std
    image += img_mean
    image = image.astype(np.uint8)
    path = os.path.join(save_dir, 'real-class={}_top-3-predictions={},{},{}.png'.format(loss_with_info[2], loss_with_info[3][2], loss_with_info[3][1], loss_with_info[3][0]))
    ski.io.imsave(path, image)