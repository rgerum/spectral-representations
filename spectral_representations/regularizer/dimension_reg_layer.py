import tensorflow as tf
from tensorflow import keras
from .spectral_slope import get_alpha, get_alpha_regularizer
import inspect


class DimensionReg(keras.layers.Layer):
    """ a layer to calculate and regularize the exponent of the eigenvalue spectrum """

    def __init__(self, strength=0.01, target_value=1, min_x=0, max_x=1000,
                 use_gamma=False, weighting=True, fix_slope=True, fit_offset=True,
                 offset=None,
                 metric_name=None, **kwargs):
        super().__init__(**kwargs)
        for item in inspect.signature(DimensionReg).parameters:
            setattr(self, item, eval(item))

        if metric_name is None:
            metric_name = self.name.replace("dimension_reg", "alpha")
        self.metric_name = metric_name
        self.calc_alpha = True

    def build(self, input_shape):
        if self.offset is not None:
            self.offset = self.add_weight(shape=(1,), name="offset", initializer="zeros", trainable=True)
        super().build(input_shape)

    def get_config(self):
        return {"strength": self.strength, "target_value": self.target_value, "metric_name": self.metric_name,
                "min_x": self.min_x, "max_x": self.max_x}

    def call(self, x):
        # get the alpha value
        if self.calc_alpha:
            # flatten the non-batch dimensions
            if self.gamma:
                data = get_alpha(x, strength=self.strength, target_alpha=self.target_value,
                                 min_x=self.min_x, max_x=self.max_x,
                                 use_gamma=self.use_gamma, weighting=self.weighting,
                                 fix_slope=True, fit_offset=True,
                                 offset=self.offset,
                                 )
                loss = data["loss"]
                mse = data["mse"]
                r2 = data["r2"]
                alpha = data["alpha"]
            else:
                loss, mse, r2 = get_alpha_regularizer(x, tau=self.min_x, N=self.max_x, alpha=self.target_value, strength=self.strength)
                alpha = 0
        else:
            alpha = 0
            mse = 0
            r2 = 0
            loss = 0
        # record it as a metric
        self.add_metric(alpha, self.metric_name)
        self.add_metric(mse, self.metric_name + "_mse")
        self.add_metric(r2, self.metric_name + "_r2")
        # calculate the loss and add is a metric
        self.add_metric(loss, self.metric_name + "_loss")
        self.add_loss(loss)

        # return the unaltered x
        return x
