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
    I(K;Y) where K to X to Y
    P_K: [K]
    Q: [K, X]
    T: [X, Y]
    Returns MI, P_X, P_Y
    """
    P_X = tf.linalg.matvec(Q, P_K, transpose_a=True)

    Q_exp = tf.expand_dims(Q, axis=2)
    T_exp = tf.expand_dims(T, axis=0)
    joint_KXY = Q_exp * T_exp

    joint_KY = tf.reduce_sum(joint_KXY, axis=1)
    P_KY = joint_KY * tf.reshape(P_K, [-1, 1])
    P_Y = tf.reduce_sum(P_KY, axis=0)

    P_KY_safe = P_KY + epsilon
    P_K_safe = tf.reshape(P_K, [-1, 1]) + epsilon
    P_Y_safe = tf.reshape(P_Y, [1, -1]) + epsilon

    MI = tf.reduce_sum(P_KY_safe * tf.math.log(P_KY_safe / (P_K_safe * P_Y_safe)))
    return MI, P_X, P_Y


def mutual_info_XY_under_PN(T, P_N, eps=EPS):
    """
    X sampled from P_N
    Y sampled from sum_x P_N[x] T[y|x]
    I(X;Y) = sum_x P_N[x] KL(T[x,:] || P_Y)
    """
    P_N = tf.convert_to_tensor(P_N, tf.float32)
    P_N = P_N / (tf.reduce_sum(P_N) + eps)
    P_Y = tf.linalg.matvec(tf.transpose(T), P_N)

    T_safe = tf.clip_by_value(T, eps, 1.0)
    P_Y_safe = tf.clip_by_value(P_Y, eps, 1.0)
    log_ratio = tf.math.log(T_safe / P_Y_safe[None, :])
    return tf.reduce_sum(P_N[:, None] * T_safe * log_ratio)



def expected_relative_cost(T, P_N, X_values, Y_values=None, epsilon=1e-8):
    """
    Expected relative cost of shaping

    E_x E_y max((y - x) / x, 0)

    T       : [Nx, Ny]  row stochastic kernel P(Y=y | X=x)
    P_N     : [Nx]      benign prior over X
    X_values: iterable, length Nx
    Y_values: iterable, length Ny, if None uses X_values (backward compatible)

    If X_values == Y_values and T is square, this reduces to the original cost
    used in your previous implementation.
    """
    T = tf.convert_to_tensor(T, dtype=tf.float32)
    P_N = tf.convert_to_tensor(P_N, dtype=tf.float32)

    X_values = tf.cast(tf.convert_to_tensor(X_values), tf.float32)
    if Y_values is None:
        Y_values = X_values
    else:
        Y_values = tf.cast(tf.convert_to_tensor(Y_values), tf.float32)

    x_sizes = tf.reshape(X_values, [-1, 1])  # [Nx, 1]
    y_sizes = tf.reshape(Y_values, [1, -1])  # [1, Ny]

    rel_delta = tf.maximum((y_sizes - x_sizes) / tf.maximum(x_sizes, epsilon), 0.0)

    expected_rel_per_x = tf.reduce_sum(T * rel_delta, axis=1)   # [Nx]
    cost = tf.reduce_sum(expected_rel_per_x * P_N)
    return cost

