''' Build linear regression from scratch with TensorFlow '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

# Set constants
N = 1000
learn_rate = 0.1
batch_size = 40
num_batches = 400

print(tf.__version__)

# Step 1: Generate input points
x = np.random.normal(size=N)

m_real = np.random.normal(loc=0.5, scale=0.2, size=N)

b_real = np.random.normal(loc=1.0, scale=0.2, size=N)
y = m_real * x + b_real

# Step 2: Create variables and placeholders
#m = tf.Variable(tf.random.normal([]))
m= tf.Variable(tf.random.normal((1, 1)))
b = tf.Variable(tf.random.normal([]))
x_holder = tf.Variable(tf.ones(dtype=tf.float32, shape=[batch_size]))
y_holder = tf.Variable(tf.ones(dtype=tf.float32, shape=[batch_size]))

print(' m ', m.numpy())
print("x_holder ", x_holder)
print("y_holder ", y_holder)
# Step 3: Define model and loss
#model = m * x_holder + b
model = m * x + b
print("model : ", model)
#loss = tf.reduce_mean(tf.pow(model - y_holder, 2))
loss = tf.reduce_mean(tf.pow(model - y, 2))

# Step 4: Create optimizer
optimizer = tf.optimizers.SGD(learn_rate)
def absloss(predicted_y, desired_y):
    return tf.reduce_mean(tf.abs(predicted_y - desired_y))

def sqrloss(predicted_y, desired_y):
    return tf.reduce_mean(tf.pow(predicted_y - desired_y,2))

def loss(predicted_y, desired_y):
    return tf.reduce_sum(tf.square(predicted_y - desired_y))

with tf.GradientTape() as t:
    abs_loss = absloss(model, y_holder)
    sqr_loss = sqrloss(model, y_holder)
    current_loss = loss(model, y_holder)
    #print(' Absolute current loss', abs_loss)
    #print(' Square current loss', sqr_loss)
    print(' current loss', current_loss)
    print(' m ',tf.Variable([[m.numpy()]]))
    print(' b ', tf.Variable([[b.numpy()]]))

    grads = t.gradient(current_loss, [tf.Variable([[m.numpy()]]),  tf.Variable([[b.numpy()]])])

    print(' Grads ', grads)

optimizer.apply_gradients(zip(grads,[tf.Variable([[m.numpy()]]),  tf.Variable([[b.numpy()]])]))

# Step 5: Execute optimizer in a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(num_batches):
        
        x_data = np.empty(batch_size)
        y_data = np.empty(batch_size)        
        for i in range(batch_size):
            index = np.random.randint(0, N)
            x_data[i] = x[index]
            y_data[i] = y[index]            

        sess.run(optimizer, feed_dict={x_holder: x_data, y_holder: y_data})
        
    print('m = ', sess.run(m))
    print('b = ', sess.run(b))
