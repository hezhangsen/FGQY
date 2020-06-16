import os
from google.colab import drive
drive.mount('/content/drive')
path = "/content/drive/My Drive"
os.chdir(path)
#!git clone https://github.com/tensorflow/examples.git
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
sys.path.insert(0, '/content/drive/My Drive/examples/tensorflow_examples/models/pix2pix')
import pix2pix
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
PATH = '/content/drive/My Drive/cycleGan/photo2monet/'
OUTPUT_CHANNELS = 3
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
def generate_images(model, test_input):
    prediction = model(test_input)
    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
model_to_be_restored = generator_g
checkpoint = tf.train.Checkpoint(generator_g=model_to_be_restored)
checkpoint.restore(tf.train.latest_checkpoint('/content/drive/My Drive/checkpoints/train'))
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
test = tf.data.Dataset.list_files(PATH + 'testC/*.jpg')
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    return image
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image
def preprocess_image_testC(image):
    image = load(image)
    image = tf.image.resize(image,(256,256))
    image = normalize(image)
    return image
test = test.map(preprocess_image_testC, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(1)
for i in test.take(2):
  generate_images(model_to_be_restored,i)
