import numpy as np
import tensorflow as tf
from mcts_4inarow import *

if __name__ == "__main__":
    tf_x = tf.placeholder(tf.float32, [None, 42]) 
    tf_y = tf.placeholder(tf.float32, [None, 8])
    tf_pi, tf_z = tf.split(tf_y, [7,1], 1)
    tf_image = tf.reshape(tf_x, [-1,6,7,1])

    num_of_kernel= 10
    w_conv1 = tf.Variable( tf.truncated_normal( [3, 3, 1, num_of_kernel], stddev=0.1))
    cov1 = tf.nn.conv2d(input=tf_image,
                        filter = w_conv1,
                        strides = [1, 1, 1,1],
                        padding = "SAME",
                        )
    b_conv1 = tf.Variable( tf.constant(0.1, shape=[10]))

    cov1_rl = tf.nn.relu(cov1 + b_conv1)
    pool1 = tf.nn.max_pool(cov1_rl,
                           ksize = [1,2,2,1],
                           strides = [1, 1 ,1,1],
                           padding = "SAME")
    #fully connected 
    result = tf.reshape(pool1,[-1,6*7*num_of_kernel])
    tf_y_pred = tf.contrib.layers.fully_connected(result, 8)
    print("\n\n\n\n")
    print(tf.shape(tf_y_pred))
    print("\n\n\n\n")
    tf_p, tf_v = tf.split(tf_y_pred, [7,1], 1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=tf_p, labels=tf_pi )
    cost_func = tf.nn.l2_loss(tf.subtract(tf_z, tf_v))
    loss = tf.reduce_mean(tf.add(cross_entropy, cost_func))
    train_step = tf.train.AdamOptimizer().minimize(loss)
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    BatchSize = 50
    trainSplit = 0.9
    trainStep = 1
    input_ = np.load("input.npy")
    output = np.load("output.npy")
    for _ in range(1000):
        sess.run(train_step, feed_dict = { tf_x:input_ , tf_y: output})
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)
#    print(sess.run(tf.nn.softmax(tf_p), feed_dict = {tf_x:input_}))#[:,7])
#    print(sess.run(tf_v, feed_dict = {tf_x:input_}))#[:,7])
