import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))

import tf_util
from transform_nets import input_transform_net
from gat_layers import attn_feature


def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):

    k = 30

    adj_matrix = tf_util.pairwise_distance(point_cloud)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    n_heads = 1
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(point_cloud, 16, nn_idx, activation=tf.nn.elu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer0', k=k, i=i, is_dist=True)
        attns.append(edge_feature)
        local_features.append(locals)
    neighbors_features = tf.concat(attns, axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud, -2), neighbors_features], axis=-1)

    locals_max_transform = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(neighbors_features, locals_max_transform, is_training, bn_decay, K=3, is_dist=True)

    point_cloud_transformed = tf.matmul(point_cloud, transform)

    adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    n_heads = 4
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(point_cloud_transformed, 16, nn_idx, activation=tf.nn.elu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer1', k=k, i=i, is_dist=True)
        attns.append(edge_feature)
        local_features.append(locals)
    neighbors_features = tf.concat(attns, axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud_transformed, -2), neighbors_features], axis=-1)

    locals_max1 = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)

    net = tf_util.conv2d(neighbors_features, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet1', bn_decay=bn_decay, is_dist=True)
    net1 = net

    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet2', bn_decay=bn_decay, is_dist=True)
    net2 = net

    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet3', bn_decay=bn_decay, is_dist=True)
    net3 = net

    adj_matrix = tf_util.pairwise_distance(net)
    nn_idx = tf_util.knn(adj_matrix, k=k)
    n_heads = 4
    attns = []
    local_features = []
    for i in range(n_heads):
        edge_feature, coefs, locals = attn_feature(net, 128, nn_idx, activation=tf.nn.elu,
                                                   in_dropout=0.6,
                                                   coef_dropout=0.6, is_training=is_training, bn_decay=bn_decay,
                                                   layer='layer2', k=k, i=i, is_dist=True)
        attns.append(edge_feature)
        local_features.append(locals)
    neighbors_features = tf.concat(attns, axis=-1)
    neighbors_features = tf.concat([tf.expand_dims(point_cloud_transformed, -2), neighbors_features], axis=-1)

    locals_max2 = tf.reduce_max(tf.concat(local_features, axis=-1), axis=-2, keep_dims=True)

    net = tf_util.conv2d(neighbors_features, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet4', bn_decay=bn_decay, is_dist=True)
    net4 = net

    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet5', bn_decay=bn_decay, is_dist=True)
    net5 = net

    net = tf_util.conv2d(net, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet6', bn_decay=bn_decay, is_dist=True)
    net6 = net


    net = tf_util.conv2d(tf.concat([net3, net6, locals_max1, locals_max2], axis=-1), 1024, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='gapnet8', bn_decay=bn_decay, is_dist=True)
    net8 = net

    out_max = tf_util.max_pool2d(net8, [num_point, 1], padding='VALID', scope='maxpool')

    one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=True, is_training=is_training,
                                          scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
    out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand, net8])


    net9 = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
    net9 = tf_util.dropout(net9, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
    net9 = tf_util.conv2d(net9, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net9 = tf_util.dropout(net9, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
    net9 = tf_util.conv2d(net9, 128, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
    net9 = tf_util.conv2d(net9, part_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None,
                          bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

    net9 = tf.reshape(net9, [batch_size, num_point, part_num])

    return net9



def get_loss(seg_pred, seg):
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg),
                                           axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)
    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res