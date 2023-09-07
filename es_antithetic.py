# Study 3: Monte Carlo Estimator, gradient estimator with antithetic pairs and scaling techniques to reduce variances.
# Function implementation of evolution strategy with antithetic pairs
# train_model_anti_norm() normalizes the rewards
# train_model_anti_sigma() scales 2N rewards by sigma_R, which is the standard deviation

def train_model_anti_norm(N, ALPHA, SIGMA, ITERATION, BATCH_SIZE, TRAIN_SIZE,
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
        V = np.random.randn(N, len(flatten_params))

        # antithetic pairs of rewards
        R = np.zeros(N)
        R_anti = np.zeros(N)

        # for every sample in the population
        for i in range(N): 
          flatten_params_es = np.add(flatten_params, SIGMA * V[i])
          flatten_params_es_anti = np.add(flatten_params, -1 * SIGMA * V[i])

          params_es = reshape_params(flatten_params_es, model_params, sess)
          params_es_anti = reshape_params(flatten_params_es_anti, model_params, sess)

          # calculate antithetic reward of the sample
          R[i] = -1 * sess.run(loss_es, 
                                feed_dict={x:batch[0], y_:batch[1], 
                                          w1_es:params_es[0], b1_es:params_es[1],
                                          w2_es:params_es[2], b2_es:params_es[3],
                                          w3_es:params_es[4], b3_es:params_es[5]})
          R_anti[i] = -1 * sess.run(loss_es, 
                                feed_dict={x:batch[0], y_:batch[1], 
                                          w1_es:params_es_anti[0], b1_es:params_es_anti[1],
                                          w2_es:params_es_anti[2], b2_es:params_es_anti[3],
                                          w3_es:params_es_anti[4], b3_es:params_es_anti[5]})         
        
        # normalize the final rewards
        R = R - R_anti
        norm_R = (R-np.mean(R))/np.std(R)
        
        flatten_params = flatten_params + ALPHA /(2*N*SIGMA)*np.dot(V.T, norm_R) 

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
    print('Epoch %d Validation Accuracy %g' % (EPOCH, test_accuracy))

    training_time = end_time - start_time
    min = training_time // 60
    sec = int(training_time % 60)
    print(f"Training Time: {min} m {sec} s" )
  
  return train_acc_es, val_acc_es, test_accuracy

def train_model_anti_sigma(N, ALPHA, SIGMA, ITERATION, BATCH_SIZE, TRAIN_SIZE,
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
        V = np.random.randn(N, len(flatten_params))

        # antithetic pairs of rewards
        R = np.zeros(N)
        R_anti = np.zeros(N)


        # for every sample in the population
        for i in range(N): 
          flatten_params_es = np.add(flatten_params, SIGMA * V[i])
          flatten_params_es_anti = np.add(flatten_params, -1* SIGMA * V[i])

          params_es = reshape_params(flatten_params_es, model_params, sess)
          params_es_anti = reshape_params(flatten_params_es_anti, model_params, sess)

          # calculate antithetic rewards of the sample
          R[i] = -1 * sess.run(loss_es, 
                                feed_dict={x:batch[0], y_:batch[1], 
                                          w1_es:params_es[0], b1_es:params_es[1],
                                          w2_es:params_es[2], b2_es:params_es[3],
                                          w3_es:params_es[4], b3_es:params_es[5]})
          R_anti[i] = -1 * sess.run(loss_es, 
                                feed_dict={x:batch[0], y_:batch[1], 
                                          w1_es:params_es_anti[0], b1_es:params_es_anti[1],
                                          w2_es:params_es_anti[2], b2_es:params_es_anti[3],
                                          w3_es:params_es_anti[4], b3_es:params_es_anti[5]})         
        
        # scale the rewards by sigma_R
        sigma_R = np.std(np.concatenate((R, R_anti), axis=0))
        R = R - R_anti
        flatten_params = flatten_params + ALPHA /(N*sigma_R)*np.dot(V.T, R) 

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
    print('Epoch %d Validation Accuracy %g' % (EPOCH, test_accuracy))

    training_time = end_time - start_time
    min = training_time // 60
    sec = int(training_time % 60)
    print(f"Training Time: {min} m {sec} s" )
  
  return train_acc_es, val_acc_es, test_accuracy