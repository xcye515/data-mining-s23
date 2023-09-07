# Study 1: ES vs Backpropagation
# Implementation of SGD optimization on MNIST classification task

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
import time
import matplotlib.pyplot as plt

# Load MNIST data set

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],784)
y_train = tf.keras.utils.to_categorical(y_train,10)
X_test = X_test.reshape(X_test.shape[0],784)
y_test = tf.keras.utils.to_categorical(y_test,10)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# hyperparameters

ALPHA = 0.001 # learning rate
BATCH_SIZE = 50
ITERATION = 6000

TRAIN_SIZE = len(X_train)
ITER_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE
EPOCH = ITERATION//ITER_PER_EPOCH + 1
iteration_num = range(0, 6001, 100)
save_path = "/content/drive/MyDrive/sgd.ckpt"

train_acc = []
val_acc = []

np.random.seed(42)


# Model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
scaling_factor = 2**125

w1 = tf.Variable(np.random.normal(scale=np.sqrt(2./784),size=[784,256]).astype(np.float32))
b1 = tf.Variable(np.zeros(256,dtype=np.float32))
h1 = tf.nn.bias_add(tf.matmul(x , w1), b1) / scaling_factor

w2 = tf.Variable(np.random.normal(scale=np.sqrt(2./256),size=[256,128]).astype(np.float32))
b2 = tf.Variable(np.zeros(128,dtype=np.float32))
h2 = tf.nn.bias_add(tf.matmul(h1, w2) , b2 / scaling_factor)  

w3 = tf.Variable(np.random.normal(scale=np.sqrt(2./128),size=[128,10]).astype(np.float32))
b3 = tf.Variable(np.zeros(10,dtype=np.float32))
y =  tf.nn.softmax(tf.nn.bias_add(tf.matmul(h2, w3), b3/ scaling_factor) * scaling_factor)

# metrics
corrections = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
acc = tf.reduce_mean(corrections)

cross_entropy_ = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
cross_entropy = tf.reduce_mean(cross_entropy_)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

# training

start_time = time.time()
with tf.Session() as sess:
  saver = tf.train.Saver()
  sess.run(tf.global_variables_initializer())
 
  ITER_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE
  EPOCH = ITERATION//ITER_PER_EPOCH + 1
  

  for e in range(EPOCH):
    train_dataset_batched = train_dataset.shuffle(TRAIN_SIZE).batch(BATCH_SIZE)
    iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset_batched)
    next_batch = iterator.get_next()
    
    for e_step in range(ITER_PER_EPOCH): 
      batch = sess.run(next_batch)
      step = e * ITER_PER_EPOCH + e_step
      if step in iteration_num:
        train_accuracy = sess.run(acc,feed_dict={x: batch[0], y_: batch[1]})
        print('Epoch %d Iteration %d, Training Accuracy %g' % (e, step, train_accuracy))
        train_acc.append(train_accuracy)

        test_accuracy = sess.run(acc,feed_dict={x: X_test, y_: y_test})
        print('Iteration %d Validation Accuracy %g' % (step, test_accuracy))
        val_acc.append(test_accuracy)
    
      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

  
  end_time = time.time()
  saver.save(sess, save_path)

  test_accuracy = sess.run(acc,feed_dict={x: X_test, y_: y_test})
  print('Epoch %d Validation Accuracy %g' % (EPOCH, test_accuracy))

  training_time = end_time - start_time
  min = training_time // 60
  sec = int(training_time % 60)
  print(f"Training Time: {min} m {sec} s" )

# visualization
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(iteration_num, train_acc, label='Training')
plt.plot(iteration_num, val_acc, label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.ylim([0, 1.0])
plt.legend()