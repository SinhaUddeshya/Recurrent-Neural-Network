import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

from loaddata import test_X, training_X, test_Y, training_Y

training_epochs = 100
n_dim = training_X.shape[1]
n_classes = 30
n_hidden_units_one = 280
n_hidden_units_two = 350
n_hidden_units_three = 280
n_hidden_units_four = 390
n_hidden_units_five = 370
n_hidden_units_six = 400
n_hidden_units_seven = 340
sd = 1/np.sqrt(n_dim)
learning_rate = 0.01
beta1=0.9
beta2=0.999
epsilon=1e-08
use_locking=False
name='Adam'

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3)

W_4 = tf.Variable(tf.random_normal([n_hidden_units_three,n_hidden_units_four], mean = 0, stddev=sd))
b_4 = tf.Variable(tf.random_normal([n_hidden_units_four], mean = 0, stddev=sd))
h_4 = tf.nn.sigmoid(tf.matmul(h_3,W_4) + b_4)

W_5 = tf.Variable(tf.random_normal([n_hidden_units_four,n_hidden_units_five], mean = 0, stddev=sd))
b_5 = tf.Variable(tf.random_normal([n_hidden_units_five], mean = 0, stddev=sd))
h_5 = tf.nn.sigmoid(tf.matmul(h_4,W_5) + b_5)

W_6 = tf.Variable(tf.random_normal([n_hidden_units_five,n_hidden_units_six], mean = 0, stddev=sd))
b_6 = tf.Variable(tf.random_normal([n_hidden_units_six], mean = 0, stddev=sd))
h_6 = tf.nn.sigmoid(tf.matmul(h_5,W_6) + b_6)

W_5 = tf.Variable(tf.random_normal([n_hidden_units_six,n_hidden_units_seven], mean = 0, stddev=sd))
b_5 = tf.Variable(tf.random_normal([n_hidden_units_seven], mean = 0, stddev=sd))
h_5 = tf.nn.sigmoid(tf.matmul(h_6,W_7) + b_7)

W = tf.Variable(tf.random_normal([n_hidden_units_three,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_6,W) + b)


cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2,epsilon,use_locking,name).minimize(cost_function)
init = tf.global_variable_initializer()

print("Adam reached...")

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None
i = 1

t0 = time.time()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        print("Epoch: ",epoch)
        print("TIme elapsed: ",(time.time()-t0))
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:training_X,Y:training_Y})
        cost_history = np.append(cost_history,cost)

        i = i + 1

        y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_X})
        y_true = sess.run(tf.argmax(test_Y,1))
        print("Test accuracy: ",round(sess.run(accuracy, feed_dict={X: test_X,Y: test_Y}),3))

fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")
print ("F-Score:", round(f,3))
