import tensorflow as tf


@tf.function
def get_alpha(data, min_x=0, max_x=1000):
    """ get the power law exponent of the PCA value distribution """
    # flatten the non-batch dimensions
    data = flatten(data)
    # get the eigenvalues of the covariance matrix
    eigen_values = get_pca_variance(data)

    # ensure that eigenvalues are slightly positive (prevents log from giving nan)
    eigen_values = tf.nn.relu(eigen_values) + 1e-8
    # get the logarithmic x and y values to fit
    y = tf.math.log(eigen_values)
    x = tf.math.log(tf.range(1, eigen_values.shape[0] + 1, 1.0, y.dtype))
    a, b = linear_fit(x[min_x:max_x], y[min_x:max_x])

    mse = tf.reduce_mean((b * x + a - y) ** 2)
    # return the negative of the slope
    return -b, mse


@tf.function
def flatten(inputs):
    """ flatten the non batch dimensions of a tensor. Works also with None as the batch dimension. """
    from tensorflow.python.framework import constant_op
    import functools
    import operator
    from tensorflow.python.ops import array_ops
    input_shape = inputs.shape
    non_batch_dims = input_shape[1:]
    last_dim = int(functools.reduce(operator.mul, non_batch_dims))
    flattened_shape = constant_op.constant([-1, last_dim])
    return array_ops.reshape(inputs, flattened_shape)


@tf.function
def get_pca_variance(data):
    """ calculate the eigenvalues of the covariance matrix """
    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    # Finding the Eigen Values and Vectors for the data
    sigma = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1)
    eigen_values, eigen_vectors = tf.linalg.eigh(sigma)

    # resort (from big to small) and normalize sum to 1
    return eigen_values[::-1] / tf.reduce_sum(eigen_values)


@tf.function
def linear_fit(x_data, y_data):
    """ calculate the linear regression fit for a list of xy points. """
    x_mean = tf.reduce_mean(x_data)
    y_mean = tf.reduce_mean(y_data)
    b = tf.reduce_sum((x_data - x_mean) * (y_data - y_mean)) / tf.reduce_sum((x_data - x_mean) ** 2)
    a = y_mean - (b * x_mean)
    return a, b
