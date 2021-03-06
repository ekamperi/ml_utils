import tensorflow as tf

def integrate_pdf(dist, x_range, y_range, n_samples=1000):
    """Calculate an approximation of the integral of the
    probability distribution function over the x and y range.
    This integral should, by definition, be very close to 1.0
    """

    # Create a 2D mesh grid
    xm, xM = (float(e) for e in x_range)
    ym, yM = (float(e) for e in y_range)
    x = tf.linspace(xm, xM, n_samples)     # x.shape = (1000,)
    y = tf.linspace(ym, yM, n_samples)     # y.shape = (1000,)
    X, Y = tf.meshgrid(x, y)               # X.shape = (1000, 1000), Y.shape = (1000, 1000)

    # Reshape the 2D mesh grid
    tXY = tf.reshape(
        tf.transpose(
            tf.stack((X,Y))),              # (2, 1000, 1000)
        (n_samples*n_samples, 2))          # (1000*1000, 2)

    # Calculate the "infinitesimal" area dA
    dA = ((xM - xm) * (yM - ym)) / (n_samples ** 2)

    # Calculate the probabilities on the reshaped (1000*1000, 2) grid
    # and calculate the result of the integral
    pdf = dist.prob(tXY)

    return tf.reduce_sum(pdf) * dA
