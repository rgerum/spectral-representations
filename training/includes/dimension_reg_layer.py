import tensorflow as tf
from tensorflow import keras
from .spectral_slope import get_alpha


class DimensionReg(keras.layers.Layer):
    """ a layer to calculate and regularize the exponent of the eigenvalue spectrum """

    def __init__(self, strength=0.01, target_value=1, min_x=0, max_x=1000, metric_name=None, **kwargs):
        super().__init__(**kwargs)
        self.strength = strength
        self.min_x = min_x
        self.max_x = max_x
        self.target_value = target_value
        if metric_name is None:
            metric_name = self.name.replace("dimension_reg", "alpha")
        self.metric_name = metric_name
        self.calc_alpha = True

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name,
                "min_x": self.min_x, "max_x": self.max_x}

    def call(self, x):
        # get the alpha value
        if self.calc_alpha:
            # flatten the non-batch dimensions
            alpha, mse = get_alpha(x, min_x=self.min_x, max_x=self.max_x)
            loss = tf.math.abs(alpha - self.target_value) * self.strength
        else:
            alpha = 0
            mse = 0
            loss = 0
        # record it as a metric
        self.add_metric(alpha, self.metric_name)
        self.add_metric(mse, self.metric_name + "_mse")
        # calculate the loss and add is a metric
        self.add_metric(loss, self.metric_name + "_loss")
        self.add_loss(loss)

        # return the unaltered x
        return x
