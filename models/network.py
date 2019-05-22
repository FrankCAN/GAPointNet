import tensorflow as tf
import numpy as np
import math
import sys
import os
from transform_nets import input_transform_net
from gat_layers import attn_feature
import tf_util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    k = 20

    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    n_heads = 1
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(point_cloud, 16, nn_idx, activation=tf.nn.elu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer0', k=k, i=i)
        attns.append(edge_feature)
        local_features.append(locals)
    neighbors_features = tf.concat(attns, axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), neighbors_features], axis=-1)

    locals_max_transform = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)


    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(neighbors_features, locals_max_transform, is_training, bn_decay, K=3)

    point_cloud_transformed = tf.matmul(point_cloud, transform)


    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    n_heads = 4
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(point_cloud_transformed, 16, nn_idx, activation=tf.nn.elu, in_dropout=0.6,
                                           coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                           layer='layer1', k=k, i=i)
        attns.append(edge_feature)
        local_features.append(locals)
    neighbors_features = tf.concat(attns, axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud_transformed, -2), neighbors_features], axis=-1)

    net = tf_util.conv2d(neighbors_features, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet1', bn_decay=bn_decay)
    net1 = net

    locals_max = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)


    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet2', bn_decay=bn_decay)
    net2 = net


    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet3', bn_decay=bn_decay)
    net3 = net


    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet4', bn_decay=bn_decay)
    net4 = net


    net = tf_util.conv2d(tf.concat([net1, net2, net3, net4, locals_max], axis=-1), 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='agg', bn_decay=bn_decay)

    net = tf.reduce_max(net, axis=1, keep_dims=True)

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)

    return classify_loss

