import tensorflow as tf
import numpy as np
from utils import *
from sage import SAGE
import networkx as nx
from load_data import load_nci

'''
parse
'''
tf.app.flags.DEFINE_integer('num_epochs', 17, 'number of epochs to train')
tf.app.flags.DEFINE_integer('labels', 1, 'number of label classes')
tf.app.flags.DEFINE_integer('graph_pad_length', 620, 'graph pad length for training')
tf.app.flags.DEFINE_integer('decay_step', 100, 'decay steps')
tf.app.flags.DEFINE_integer('cv_index', 2, 'fold_ID')
tf.app.flags.DEFINE_float('learn_rate', 1e-2, 'learn rate for training optimization')
tf.app.flags.DEFINE_boolean('train', False, 'train mode FLAG')

FLAGS = tf.app.flags.FLAGS

cv_index = FLAGS.cv_index
num_epochs = FLAGS.num_epochs
tag_size = FLAGS.labels
graph_pad_length = FLAGS.graph_pad_length
feature_dimension = 3
lr = FLAGS.learn_rate
fully_connected_n = 256
validation_length = 200

def fold_cv(cv_index = 2,if_train = FLAGS.train):
  placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
    'features':tf.sparse_placeholder(tf.float32, shape=tf.constant((graph_pad_length,feature_dimension), dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=()),
    'dropout': tf.placeholder_with_default(0.3, shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
  }
  # construct graph for graph classification
  with tf.Session() as sess:
    model = SAGE()
    model.build_graph(n=graph_pad_length,placeholders = placeholders,d =feature_dimension)
    with tf.variable_scope('DownstreamApplication'):
      global_step = tf.Variable(0, trainable=False, name='global_step')
      learn_rate = tf.train.exponential_decay(lr, global_step, FLAGS.decay_step, 0.98, staircase=True)
      labels = placeholders['labels']
      num_inputs = model.M.get_shape()[0:3].num_elements()
      layer_flat = tf.reshape(model.M,[1,-1])
      fc_weights = tf.Variable(tf.truncated_normal((num_inputs,fully_connected_n),stddev = 0.05))
      fc_weights = tf.nn.dropout(fc_weights, 1-placeholders['dropout'])
      biases = tf.get_variable("b", [fully_connected_n], initializer=tf.keras.initializers.he_normal(seed = 123))
      net = tf.matmul(layer_flat,fc_weights) + biases
      net = tf.nn.relu(net)
      net = tf.reshape(net,[-1])
      last_weights = tf.Variable(tf.truncated_normal((fully_connected_n,tag_size),stddev = 0.05))
      last_weights = tf.reshape(last_weights,[-1])
      logits = tf.reduce_sum(tf.multiply(net,last_weights))
      loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits) 
    
      p_coef = 0.19
      weight_decay = 0.0005
      for var in model.GCN_layer1.vars.values():
          loss += weight_decay * tf.nn.l2_loss(var)
      p_loss = p_coef * model.P
      loss = loss + p_loss
      p_loss = tf.reduce_mean(p_loss)
      params = tf.trainable_variables()
      optimizer = tf.train.AdamOptimizer(learn_rate)
      grad_and_vars = tf.gradients(loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(grad_and_vars, 0.5)
      opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
  

    # load data
    train_structure_input,train_feature_input,test_structure_input,test_feature_input,ally,ty = load_nci(cv_index)
    # graph padding
    train_structure_input,train_feature_input = graph_padding(train_structure_input,train_feature_input,graph_pad_length)
    test_structure_input,test_feature_input = graph_padding(test_structure_input,test_feature_input,graph_pad_length)
    # split train into train and validation set
    vtrain_feature_input = train_feature_input[-validation_length:]
    vtrain_structure_input = train_structure_input[-validation_length:]
    vally = ally[-validation_length:]
    train_feature_input = train_feature_input[:-validation_length]
    train_structure_input = train_structure_input[:-validation_length]
    ally = ally[:-validation_length]
    # reorder and sampletrain set into balanced set 
    train_structure_input,train_feature_input,ally = re_order(train_structure_input,train_feature_input,ally)
    total = len(train_feature_input)
    vtotal = len(vtrain_feature_input)
  
    sess.run(tf.global_variables_initializer())
    if if_train == True:
      #print('start training')
      hist_loss = []
      for epoch_num in range(num_epochs):
        epoch_loss = 0
        step_loss = 0
        idx = np.random.RandomState( seed = epoch_num + 21).permutation(total)
        for i in range(int(total)):
          batch_input,batch_topo, batch_tags = (train_feature_input[idx[i]],train_structure_input[idx[i]], ally[idx[i]])
          batch_input = preprocess_features(batch_input.tolil())
          batch_topo = [preprocess_adj(batch_topo)]
          train_ops = [opt, loss, learn_rate, global_step]
          train_ops += [p_loss]
          feed_dict = construct_feed_dict(batch_input, batch_topo, batch_tags, placeholders)
          result = sess.run(train_ops, feed_dict=feed_dict)
          step_loss += result[1]
          epoch_loss += result[1]
          step_loss = 0
        vepoch_loss = 0
        for i in range(vtotal):
          batch_input,batch_topo, batch_tags = (vtrain_feature_input[i],vtrain_structure_input[i], vally[i])
          batch_input = preprocess_features(batch_input.tolil())
          batch_topo = [preprocess_adj(batch_topo)]
          feed_dict = construct_feed_dict(batch_input, batch_topo, batch_tags, placeholders)
          feed_dict.update({placeholders['dropout']: 0})
          result = sess.run(loss, feed_dict=feed_dict)
          vepoch_loss += result
        hist_loss.append(vepoch_loss/vtotal)
        if epoch_num == 5:
          saver = tf.train.Saver()
          saver.save(sess, "./pretrained/{}/model.ckpt".format(cv_index))
        if epoch_num > 5 and hist_loss[-1] - hist_loss[-2] < 0.:
          saver = tf.train.Saver()
          saver.save(sess, "./pretrained/{}/model.ckpt".format(cv_index))
        if epoch_num > 5 and hist_loss[-1] - hist_loss[-2] > 0.:
          learn_rate = learn_rate/2.0
          saver = tf.train.Saver()
          saver.restore(sess, "./pretrained/{}/model.ckpt".format(cv_index))
      saver = tf.train.Saver()
      saver.restore(sess, "./pretrained/{}/model.ckpt".format(cv_index))
      # change optimizer from Adam to SGD for better generalization
      learn_rate = learn_rate * 10.0
      optimizer = tf.train.GradientDescentOptimizer(learn_rate)
      grad_and_vars = tf.gradients(loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(grad_and_vars, 0.5)
      opt = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
      for i in range(vtotal):
        batch_input,batch_topo, batch_tags = (vtrain_feature_input[i],vtrain_structure_input[i], vally[i])
        batch_input = preprocess_features(batch_input.tolil())
        batch_topo = [preprocess_adj(batch_topo)]
        feed_dict = construct_feed_dict(batch_input, batch_topo, batch_tags, placeholders)
        train_ops = [opt, loss, learn_rate, global_step]
        result = sess.run(train_ops, feed_dict=feed_dict)
      saver = tf.train.Saver()
      saver.save(sess, "./pretrained/{}/model.ckpt".format(cv_index))
    else:
      saver = tf.train.Saver()
      saver.restore(sess, "./pretrained/{}/model.ckpt".format(cv_index))

    total = len(test_feature_input)

    RESULT = []
    #print('start testing')
    for i in range(total):
      batch_input,batch_topo, batch_tags = (test_feature_input[i],test_structure_input[i], ty[i])
      batch_input = preprocess_features(batch_input.tolil())
      batch_topo = [preprocess_adj(batch_topo)]
      feed_dict = construct_feed_dict(batch_input, batch_topo, batch_tags, placeholders)
      feed_dict.update({placeholders['dropout']: 0})
      result = sess.run([tf.nn.sigmoid(logits)], feed_dict=feed_dict)
      RESULT.append(result[0])
    prediction = np.asarray(RESULT)
    y_test = np.asarray(ty)

    predictions = np.asarray([0 if i<0.5 else 1 for i in prediction.tolist()]).astype(int)
    correct_prediction = np.equal(predictions, y_test)
    sess.close()
    return np.sum(correct_prediction)/float(total)
