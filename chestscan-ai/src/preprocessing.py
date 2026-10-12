# Image preprocessing utilities for ChestScan AI.

import tensorflow as tf

IMG_SIZE = (180, 180)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def prepare_image(image):
    """Decode a file path or raw bytes into an RGB float32 tensor."""
    if isinstance(image, str):
        image = tf.io.read_file(image)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    return tf.cast(image, tf.float32)
