import tensorflow as tf

@tf.function
def log10(x):
    x1 = tf.math.log(x)
    x2 = tf.math.log(10.0)
    return x1 / x2

@tf.function
def get_alpha(data, min_x=0, max_x=1000, target_alpha=1, strength=0, clip_pred_y=True, offset=None):
    """ get the power law exponent of the PCA value distribution """
    # flatten the non-batch dimensions
    data = flatten(data)
    # get the eigenvalues of the covariance matrix
    eigen_values = get_pca_variance(data)

    if max_x == -1:
        max_x = tf.cast(data.shape[1] / 2, tf.int32)
        print("dynamic max_x", max_x)

    # ensure that eigenvalues are slightly positive (prevents log from giving nan)
    eigen_values = tf.nn.relu(eigen_values) + 1e-8
    # get the logarithmic x and y values to fit
    y = log10(eigen_values)
    x = log10(tf.range(1, eigen_values.shape[0] + 1, 1.0, y.dtype))

    # constraint with min_x and max_x
    y2 = y[min_x:max_x]
    x2 = x[min_x:max_x]

    if 0:  # no weighting
        weights = x2*0+1
    else:  # logarithmic weighting
        weights = x[1:] - x[:-1]
        weights = weights[min_x:max_x]

    if 1:  # fix slope
        m2 = -target_alpha
        if offset is None:
            if 1:  # fit offset
                t2 = fit_offset_w(x2, y2, m2, weights)
            else:  # fix offset
                t2 = y2[min_x] - m2 * x2[min_x]
        else:
            t2 = offset
    else:  # fit slope
        t2, m2 = linear_fit(x2, y2)

    pred_y = m2 * x2 + t2
    if clip_pred_y is True:
        pred_y = tf.clip_by_value(pred_y, clip_value_min=tf.reduce_min(y2), clip_value_max=tf.reduce_max(y2))
    mse = tf.reduce_sum((pred_y - y2) ** 2 * weights) / tf.reduce_sum(weights)

    r2 = get_r2(y2, m2 * x2 + t2)

    loss = strength * mse

    t, m = linear_fit(x2, y2)

    # return the negative of the slope
    return loss, -m, mse, r2, x, y, (t, m, t2, m2)

@tf.function
def get_alpha_regularizer(data, tau=5, N=1000, alpha=1., strength=1):
    # flatten the non-batch dimensions
    data = flatten(data)

    lambdas = get_pca_variance(data)
    lambdas = tf.cast(lambdas, tf.float32)
    #N = lambdas.shape[0]
    lambdas = lambdas[tau:N]
    kappa = lambdas[0] * tf.math.pow(float(tau), alpha)
    gammas = kappa * tf.math.pow(tf.range(tau, N, dtype=tf.float32), -alpha)
    loss = 1/N * tf.reduce_sum((lambdas/gammas - 1) ** 2 + tf.nn.relu(lambdas/gammas - 1))

    mse = tf.reduce_mean((gammas - lambdas) ** 2)
    r2 = get_r2(lambdas, gammas)

    return loss, mse, r2

@tf.function
def get_r2(y, y_pred):
    ss_res = tf.reduce_sum((y_pred - y) ** 2)
    ss_tot = tf.reduce_sum((y - tf.reduce_mean(y)) ** 2)
    return 1 - ss_res / ss_tot

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
    m = tf.reduce_sum((x_data - x_mean) * (y_data - y_mean)) / tf.reduce_sum((x_data - x_mean) ** 2)
    t = y_mean - (m * x_mean)
    return t, m

@tf.function
def fit_offset(x_data, y_data, m):
    """ calculate the linear regression fit for a list of xy points. """
    x_mean = tf.reduce_mean(x_data)
    y_mean = tf.reduce_mean(y_data)
    t = y_mean - (m * x_mean)
    return t

@tf.function
def fit_offset_w(x_data, y_data, m, w):
    """ calculate the linear regression fit for a list of xy points. """
    x_mean = tf.reduce_sum(x_data*w)/tf.reduce_sum(w)
    y_mean = tf.reduce_sum(y_data*w)/tf.reduce_sum(w)
    t = y_mean - (m * x_mean)
    return t
