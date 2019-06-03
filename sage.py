import tensorflow as tf
from layers import *


class SAGE(object):
  def build_graph(self, placeholders, n=100, d=3, hidden_d=128,u=16, d_a=64, r=16,reuse=False):
    with tf.variable_scope('SelfAttentiveGraph', reuse=reuse):
      self.n = n
      self.d = d
      self.d_a = d_a
      self.u = u
      self.r = r

      initializer = tf.keras.initializers.he_normal(seed = 123)

      self.input_F = placeholders['features']
      self.placeholders = placeholders
      self.GCN_layer1 = GraphConvolution(input_dim=self.d,output_dim=hidden_d,placeholders=self.placeholders,act=tf.nn.relu,dropout=True,sparse_inputs=True,logging=True)
      hidden = self.GCN_layer1(self.input_F)
      self.GCN_layer2 = GraphConvolution(input_dim=hidden_d,output_dim=self.u,placeholders=self.placeholders,act=tf.nn.relu,dropout=True,logging=False)
      self.H = self.GCN_layer2(hidden)
      
       
      self.W_s1 = tf.get_variable('W_s1', shape=[self.d_a, self.u],
          initializer=initializer)
      self.W_s2 = tf.get_variable('W_s2', shape=[self.r, self.d_a],
          initializer=initializer)

      self.batch_size = batch_size = tf.shape(self.input_F)[0]
      A = self.A = tf.nn.softmax(tf.matmul(self.W_s2, tf.tanh(tf.matmul(self.W_s1, tf.transpose(self.H)))))
      self.M = tf.matmul(A, self.H)
      A_T = tf.transpose(A, perm=[1, 0])
      AA_T = tf.matmul(A, A_T) - tf.eye(r)
      self.P = tf.square(tf.norm(AA_T))

  def trainable_vars(self):
    return [var for var in
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='SelfAttentiveGraph')]
