
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import models,layers
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML




BATCH_SIZE = 32
IMAGE_SIZE = (256,256)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=24
)

