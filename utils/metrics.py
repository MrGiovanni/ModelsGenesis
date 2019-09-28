import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_mean_iou


def sparse_accuracy_ignoring_last_label(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, nb_classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       nb_classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[-1], tf.bool)
    y_true = tf.stack(unpacked[:-1], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

# This IOU implementation is wrong!!!
'''def mean_iou_ignoring_last_label(y_true, y_pred):
    batch_size = K.int_shape(y_pred)[0]
    y_true_list = tf.unpack(y_true, num=batch_size, axis=0)
    y_pred_list = tf.unpack(y_pred, num=batch_size, axis=0)
    mean_iou = 0.
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        nb_classes = K.int_shape(y_pred)[-1]
        y_pred = K.reshape(y_pred, (-1, nb_classes))
        y_pred = K.argmax(y_pred, axis=-1)
        y_pred = K.one_hot(y_pred, nb_classes)
        y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), nb_classes + 1)
        unpacked = tf.unpack(y_true, axis=-1)
        legal_labels = tf.expand_dims(tf.to_float(
            ~tf.cast(unpacked[-1], tf.bool)), -1)
        y_true = tf.pack(unpacked[:-1], axis=-1)
        y_true = K.argmax(y_true, axis=-1)
        y_true = K.one_hot(y_true, nb_classes)
        y_pred = tf.cast(y_pred, tf.bool)
        y_true = tf.cast(y_true, tf.bool)

        intersection = tf.to_float(y_pred & y_true) * legal_labels
        union = tf.to_float(y_pred | y_true) * legal_labels
        intersection = K.sum(intersection, axis=0)
        union = K.sum(union, axis=0)
        total_union = K.sum(tf.to_float(tf.cast(union, tf.bool)))
        iou = K.sum(intersection / (union + K.epsilon())) / total_union
        mean_iou = mean_iou + iou
    mean_iou = mean_iou / batch_size
    return mean_iou'''
