import numpy as np
import copy
import time
import tensorflow as tf

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def _policy(state):
    pred = sess.run(tf_y_pred, feed_dict = {tf_x: state}).reshape(-1)
#    p = softmax(pred[:7])
#    v = np.tanh(pred[7])
#        v = np.random.rand()
#        p = np.random.dirichlet([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return pred
                    
def cnn_layer(x, kernel):
    w_conv = tf.Variable(tf.truncated_normal(kernel, stddev=0.1))
    conv = tf.nn.conv2d(input = x,
                        filter = w_conv,
                        strides = [1, 1, 1, 1],
                        padding = "SAME"
    )
    b_conv = tf.Variable(tf.constant(0.1, shape=[kernel[3]]))

    conv_rl = tf.nn.relu(conv + b_conv)
    return conv_rl


if __name__=="__main__":
    tf_x = tf.placeholder(tf.float32, [None, 42])
    tf_y = tf.placeholder(tf.float32, [None, 8])
    tf_pi, tf_z = tf.split(tf_y, [7,1], 1)
    tf_image = tf.reshape(tf_x, [-1,6,7,1])
    
    array_kernel = [[3, 3, 1, 32], [3, 3, 32, 64], [3, 3, 64, 1024]]
    conv = tf_image
    for kernel in array_kernel:
        conv = cnn_layer(conv, kernel)

    #fully connected
    result = tf.reshape(conv,[-1,6*7*1024])
    tf_y_pred = tf.contrib.layers.fully_connected(result, 8, activation_fn = None)
    tf_p, tf_v = tf.split(tf_y_pred, [7,1], 1)

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/model.ckpt")

    a = np.load("input.npy")
    print(a[-40].reshape(6, 7))
    print(_policy(a[-40].reshape(1, 42)))
