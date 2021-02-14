import tensorflow as tf

@tf.function
def nll(dist, x_train):
    """Calculates the negative log-likelihood for a given distribution
    and a data set.
    """
    return -tf.reduce_mean(dist.log_prob(x_train))


@tf.function
def get_loss_and_grads(dist, x_train):
    """Returns a tuple of (loss, gradients) for a given distribution
    and a dat set. The loss is the negative log-likelihood.
    """
    with tf.GradientTape() as tape:
        tape.watch(dist.trainable_variables)
        loss = nll(dist, x_train)
        grads = tape.gradient(loss, dist.trainable_variables)
    return loss, grads
