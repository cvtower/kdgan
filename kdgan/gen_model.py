from kdgan import config

from nets import nets_factory

import numpy as np
import tensorflow as tf

class GEN():
    def __init__(self, model_name,
            num_label=config.num_label,
            l2_coefficient=0.00004,
            is_training=True,
            preprocessing_name=None):
        self.is_training = is_training
        self.preprocessing_name = preprocessing_name or model_name

        network_fn = nets_factory.get_network_fn(model_name,
                num_classes=num_label,
                weight_decay=l2_coefficient,
                is_training=is_training)
        image_size = network_fn.default_image_size
        self.image_size = image_size

        self.images_ph = tf.placeholder(tf.float32,
                shape=(None, image_size, image_size, config.channels))
        self.labels_ph = tf.placeholder(tf.float32,
                shape=(None, num_label))





