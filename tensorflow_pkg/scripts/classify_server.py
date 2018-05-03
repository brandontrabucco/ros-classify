#!/usr/bin/env python

from tensorflow_pkg.srv import Model, ModelResponse
import tensorflow as tf
import numpy as np
import rospy

def handle_classify(req):

    print("Running classification on:")
    print(req.input_array)

    with tf.Graph().as_default():

        weights = tf.get_variable(
            "weights",
            [16, 3],
            initializer=tf.truncated_normal_initializer(
                stddev=0.1,
                dtype=tf.float32),
            dtype=tf.float32)
        biases = tf.get_variable(
            "biases",
            [3],
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)

        tensor_in = tf.placeholder(tf.float32, shape=(16))
        tensor_out = tf.argmax(
            tf.nn.softmax(
                tf.tensordot(
                    tensor_in,
                    weights,
                    1
                ) + biases))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return ModelResponse(
                sess.run(
                    tensor_out, 
                    feed_dict={
                        tensor_in: req.input_array}))

def classify_server():
    rospy.init_node('classify_server')
    rospy.Service('classify', Model, handle_classify)
    print("Ready to classify float32[16] array.")
    rospy.spin()

if __name__ == "__main__":
    classify_server()
