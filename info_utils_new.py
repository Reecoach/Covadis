# -*- coding: utf-8 -*-
"""
Information and cost utilities
"""

from __future__ import annotations
import tensorflow as tf

EPS = 1e-10


def kl_divergence(P, Q, epsilon=EPS):
    P = tf.clip_by_value(P, epsilon, 1.0)
    Q = tf.clip_by_value(Q, epsilon, 1.0)
    return tf.reduce_sum(P * tf.math.log(P / Q))


def entropy_vec(P, epsilon=EPS):
    P = tf.clip_by_value(P, epsilon, 1.0)
    return -tf.reduce_sum(P * tf.math.log(P))


def row_entropy(P, epsilon=EPS):
    P = tf.clip_by_value(P, epsilon, 1.0)
    H = -tf.reduce_sum(P * tf.math.log(P), axis=1)
    return tf.reduce_mean(H)


def col_entropy(P, epsilon=EPS):
    col_sum = tf.reduce_sum(P, axis=0, keepdims=True) + epsilon
    col_dist = P / col_sum
    H = -tf.reduce_sum(col_dist * tf.math.log(col_dist), axis=0)
    return tf.reduce_mean(H)


def compute_mutual_info(P_K, Q, T, epsilon=EPS):
    """
    I(K;Y) with K -> X -> Y
    Q: [K, X]
    T: [X, Y]
    """
    P_X = tf.linalg.matvec(Q, P_K, transpose_a=True)

    joint_KXY = tf.expand_dims(Q, 2) * tf.expand_dims(T, 0)
    joint_KY = tf.reduce_sum(joint_KXY, axis=1)

    P_KY = joint_KY * tf.reshape(P_K, [-1, 1])
    P_Y = tf.reduce_sum(P_KY, axis=0)

    P_KY_safe = P_KY + epsilon
    P_K_safe = tf.reshape(P_K, [-1, 1]) + epsilon
    P_Y_safe = tf.reshape(P_Y, [1, -1]) + epsilon

    MI = tf.reduce_sum(
        P_KY_safe * tf.math.log(P_KY_safe / (P_K_safe * P_Y_safe))
    )
    return MI, P_X, P_Y


def mutual_info_XY_under_PN(T, P_N, eps=EPS):
    """
    X ~ P_N, Y ~ T(X)
    """
    P_N = tf.convert_to_tensor(P_N, tf.float32)
    P_N = P_N / (tf.reduce_sum(P_N) + eps)

    P_Y = tf.linalg.matvec(tf.transpose(T), P_N)

    T_safe = tf.clip_by_value(T, eps, 1.0)
    P_Y_safe = tf.clip_by_value(P_Y, eps, 1.0)

    log_ratio = tf.math.log(T_safe / P_Y_safe[None, :])
    return tf.reduce_sum(P_N[:, None] * T_safe * log_ratio)


def expected_relative_cost_xy(T, P_N_X, values_X, values_Y, epsilon=1e-8):
    """
    General cost: X -> Y with different domains
    """
    T = tf.convert_to_tensor(T, tf.float32)
    P_N_X = tf.convert_to_tensor(P_N_X, tf.float32)

    x_vals = tf.reshape(tf.cast(values_X, tf.float32), [-1, 1])
    y_vals = tf.reshape(tf.cast(values_Y, tf.float32), [1, -1])

    rel = tf.maximum((y_vals - x_vals) / tf.maximum(x_vals, epsilon), 0.0)
    expected = tf.reduce_sum(T * rel, axis=1)
    return tf.reduce_sum(expected * P_N_X)
