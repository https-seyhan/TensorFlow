from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.chdir('/home/saul/Business')
# Step 1: Create a TFRecordDataset and iterator
dset = tf.data.TFRecordDataset('mnist_test.tfrecords')
iter = dset.__iter__()   #.make_one_shot_iterator()

# Step 2: Create a dictionary that describes the examples
feature_dict = {'images': tf.io.FixedLenFeature([], tf.string),
                'labels': tf.io.FixedLenFeature([], tf.int64)}

# Step 3: Parse the first example

example = iter.get_next()
mnist = tf.io.parse_single_example(example, feature_dict)
    
# Step 4: Convert the data in the first image to a 28-by-28 array
pixels = tf.io.decode_raw(mnist['images'], tf.uint8)
print("Pixels ", pixels)
pixel_matrix = tf.reshape(pixels, [28, 28])
    
# Step 5: Display the image
plt.imshow(pixel_matrix, cmap='gray')
plt.show()