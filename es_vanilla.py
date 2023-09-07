# Complete implementation of Vanilla Evolution Strategy Optimization on MNIST Classification Task
# Base model of es optimization for all study 1-3

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

N = 10 # population size
SIGMA = 0.002 # noise standard deviation
ALPHA = 0.001 # learning rate
BATCH_SIZE = 50
TRAIN_SIZE = len(X_train)
ITERATION = 6000
iteration_num = range(0, 6001, 100)
save_path = "/content/drive/MyDrive/"

np.random.seed(42)


# Evolution Strategy
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
y =  tf.nn.bias_add(tf.matmul(h2, w3), b3/ scaling_factor) * scaling_factor

model_params = [w1, b1, w2, b2, w3, b3]

w1_es =  tf.placeholder(tf.float32,[784,256])
b1_es = tf.placeholder(tf.float32,256)
h1_es = tf.nn.bias_add(tf.matmul(x , w1_es), b1_es) / scaling_factor

w2_es = tf.placeholder(tf.float32,[256,128])
b2_es = tf.placeholder(tf.float32,128)
h2_es = tf.nn.bias_add(tf.matmul(h1_es, w2_es) , b2_es / scaling_factor)  

w3_es = tf.placeholder(tf.float32,[128,10])
b3_es = tf.placeholder(tf.float32,10)
reward =  tf.nn.bias_add(tf.matmul(h2_es, w3_es), b3_es / scaling_factor) * scaling_factor

w1_new = tf.placeholder(tf.float32,[784,256])
b1_new = tf.placeholder(tf.float32,256)
w2_new = tf.placeholder(tf.float32,[256,128])
b2_new = tf.placeholder(tf.float32,128)
w3_new = tf.placeholder(tf.float32,[128,10])
b3_new = tf.placeholder(tf.float32,10)

update1 = tf.assign(w1,w1_new)
update2 = tf.assign(b1,b1_new)
update3 = tf.assign(w2,w2_new)
update4 = tf.assign(b2,b2_new)
update5 = tf.assign(w3,w3_new)
update6 = tf.assign(b3,b3_new)

# aux functions
def cross_entropy(y1_, y2):
  ce = tf.nn.softmax_cross_entropy_with_logits(labels=y1_, logits=y2)
  ce = tf.reduce_mean(ce)
  return ce

# loss functions
loss_train = cross_entropy(y_, y)
loss_es = cross_entropy(y_, reward)

# accuracy
corrections = tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32)
acc = tf.reduce_mean(corrections)

# reshape params
def reshape_params(_flatten_params, _model_params, sess):
  params_reshaped = []
  start = 0
  for _param in _model_params:
    p = sess.run(_param)
    if len(p.shape) == 1:
      nrow = p.shape[0]
      ncol = 1
    if len(p.shape) == 2:
      nrow, ncol = p.shape
              
    end = start + nrow * ncol
    params_layer = _flatten_params[start:end]
    params_layer = np.reshape(params_layer, p.shape)
    params_reshaped.append(params_layer)
    
    start = end
  
  return params_reshaped

def train_model(N, ALPHA, SIGMA, ITERATION, BATCH_SIZE, TRAIN_SIZE,
                iteration_num, train_dataset, X_test, y_test,
                save_path):
  
  ITER_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE
  EPOCH = ITERATION//ITER_PER_EPOCH + 1
  val_acc_es = []
  train_acc_es = []
  
  start_time = time.time()

  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # convert the train dataset to mini batches

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
          train_acc_es.append(train_accuracy)

          test_accuracy = sess.run(acc,feed_dict={x: X_test, y_: y_test})
          print('Iteration %d Validation Accuracy %g' % (step, test_accuracy))
          val_acc_es.append(test_accuracy)
        
      
        flatten_params = []
        for param in model_params:
          p = sess.run(param)
          k = p.flatten()
          flatten_params = np.concatenate((flatten_params,k),axis=0)
          
        # get all model parameters
        flatten_params = np.reshape(flatten_params,(1,len(flatten_params)))[0]

        # noise vector V, number of population N x number of parameter n_params
        V = np.random.randn(N, len(flatten_params))*SIGMA

        # reward
        R = np.zeros(N)

        # for every sample in the population
        for i in range(N): 
          flatten_params_es = np.add(flatten_params, V[i])
          params_es = reshape_params(flatten_params_es, model_params, sess)
          # calculate reward of the sample
          R[i] = -1 * sess.run(loss_es, 
                                feed_dict={x:batch[0], y_:batch[1], 
                                          w1_es:params_es[0], b1_es:params_es[1],
                                          w2_es:params_es[2], b2_es:params_es[3],
                                          w3_es:params_es[4], b3_es:params_es[5]})
            
        # raw reward
        norm_R = (R - np.mean(R)) / np.std(R)
        flatten_params = flatten_params + ALPHA /(N*(SIGMA**2))*np.dot(V.T, norm_R)

        # convert flatten parameters to input shape
        params_new = reshape_params(flatten_params, model_params, sess)          

        sess.run(update1,feed_dict={w1_new:params_new[0]})
        sess.run(update2,feed_dict={b1_new:params_new[1]})
        sess.run(update3,feed_dict={w2_new:params_new[2]})
        sess.run(update4,feed_dict={b2_new:params_new[3]})
        sess.run(update5,feed_dict={w3_new:params_new[4]})
        sess.run(update6,feed_dict={b3_new:params_new[5]})
    
    end_time = time.time()

    path = saver.save(sess, save_path)
    print ("Model saved in file: ", path)

    test_accuracy = sess.run(acc,feed_dict={x: X_test, y_: y_test})
    print('Epoch %d Finished; Validation Accuracy %g' % (EPOCH, test_accuracy))

    training_time = end_time - start_time
    min = training_time // 60
    sec = int(training_time % 60)
    print(f"Training Time: {min} m {sec} s" )
  
  return train_acc_es, val_acc_es, test_accuracy

# get model performances
train_acc_es, val_acc_es, test_accuracy = train_model(
                N, ALPHA, SIGMA, ITERATION, BATCH_SIZE, TRAIN_SIZE,
                iteration_num, train_dataset, X_test, y_test,
                save_path)

# visualization
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(iteration_num, train_acc_es, label='Training')
plt.plot(iteration_num, val_acc_es, label='Validation')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.ylim([0, 1.0])
plt.legend()
