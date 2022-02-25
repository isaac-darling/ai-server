import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

#comm = np.load("comm.npz")
with np.load("img.npz") as f:
    image = list(f.values())[0][170]

image = tf.expand_dims(image, 0)
image = tf.image.crop_to_bounding_box(image, 30, 0, 36, 128)
image = tf.transpose(image, [0, 3, 1, 2])

#print(image.shape)
tf.keras.preprocessing.image.save_img("test.jpg", image)