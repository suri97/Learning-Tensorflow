import tensorflow as tf
import load_model

learning_rate = 0.001
num_epochs = 100
display_step = 5

num_intput = 9
num_output = 1

layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, num_intput))

# Layer 1
with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weights_1", shape=[num_intput, layer_1_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases_1", shape=[layer_1_nodes], initializer=tf.zeros_initializer())
    layer_1_output = tf.matmul(X, weights) + biases
    layer_1_output = tf.nn.relu(layer_1_output)

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights_2", shape=[layer_1_nodes, layer_2_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases_2", shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.matmul(layer_1_output, weights) + biases
    layer_2_output = tf.nn.relu(layer_2_output)

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights_3", shape=[layer_2_nodes, layer_3_nodes],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases_3", shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.matmul(layer_2_output, weights) + biases
    layer_3_output = tf.nn.relu(layer_3_output)

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights_4", shape=[layer_3_nodes, num_output],
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases_4", shape=[num_output], initializer=tf.zeros_initializer())

    prediction = tf.matmul(layer_3_output, weights) + biases
    prediction = tf.nn.relu(prediction)

# Cost
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, num_output))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))


with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

saver = tf.train.Saver()

#init = tf.global_variables_initializer()

with tf.Session() as sess:
    #sess.run(init)

    saver.restore(sess, './logs/trained_model.ckpt')

    final_train_cost = sess.run(cost, feed_dict={
        X: load_model.data['X_train'],
        Y: load_model.data['Y_train']
    })

    final_test_cost = sess.run(cost, feed_dict={
        X: load_model.data['X_test'],
        Y: load_model.data['Y_test']
    })

print ("Final Training Cost is ", final_train_cost)
print ("Final Testing Cost is ", final_test_cost)


