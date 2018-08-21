import numpy as np
import tensorflow as tf
from mcts_4inarow import *

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

if __name__ == "__main__":
    tf_x = tf.placeholder(tf.float32, [None, 42]) 
    tf_y = tf.placeholder(tf.float32, [None, 8])
    tf_pi, tf_z = tf.split(tf_y, [7, 1], 1)
    tf_image = tf.reshape(tf_x, [-1, 6, 7, 1])

    array_kernel = [[3, 3, 1, 32], [3, 3, 32, 64], [3, 3, 64, 1024]]
    conv = tf_image
    for kernel in array_kernel:
        conv = cnn_layer(conv, kernel)
    #fully connected 
    result = tf.reshape(conv ,[-1, 6 * 7 * 1024])
    tf_y_pred = tf.contrib.layers.fully_connected(result, 8, activation_fn = None)
    """
    tf_p, tf_v0 = tf.split(tf_y_pred, [7,1], 1)
    tf_v = tf.tanh(tf_v0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=tf_p, labels=tf_pi )
    cost_func = tf.nn.l2_loss(tf.subtract(tf_z, tf_v))
    loss = tf.reduce_mean(tf.add(cross_entropy, cost_func))
    train_step = tf.train.AdamOptimizer().minimize(loss)
    """
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/model.ckpt")
    
    BatchSize = 50
    trainSplit = 0.9
    trainStep = 1
    input_ = np.load("input.npy")[-50000:]
    output = np.load("output.npy")[-50000:]

    for _ in range(1000):
        in_, out = generate_batch(input_, output, BatchSize)
        sess.run(train_step, feed_dict = {tf_x: in_, tf_y: out})
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
    print(sess.run(tf.nn.softmax(tf_p), feed_dict = {tf_x:input_[-20:]}))#[:,7])
    print(sess.run(tf_v, feed_dict = {tf_x:input_[-20:]}))#[:,7])
