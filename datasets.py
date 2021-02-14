import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def make_checkerboard(x_range, y_range, block_size=(10,10), n_samples=100, add_noise=False):
    """Generates a checkerboard distribution spanning x_range
    across the x axis and y_range across the y axis. The
    block_size is the size of the checker pattern that is tiled
    and n_samples is the number of points of the initial grid
    along each axis.

    board = make_checkerboard((-1, 1), (-1, 1), block_size=(11,11), n_samples=80, add_noise=True)
    plt.figure(figsize=(6,6))
    plt.scatter(board[:,0], board[:,1], s=2);
    """
    xm, xM = (float(e) for e in x_range)
    ym, yM = (float(e) for e in y_range)
    x = np.linspace(xm, xM, n_samples)     # x.shape = (100,)
    y = np.linspace(ym, yM, n_samples)     # y.shape = (100,)
    X, Y = np.meshgrid(x, y)               # X.shape = (100, 100), Y.shape = (100, 100)
    XY = np.transpose(
            np.stack((X,Y)))               # XY.shape = (100, 100, 2)

    # Calculate the checkboard's dimensions
    block_nx, block_ny = block_size
    board_nx = math.ceil((float(n_samples)) / (2 * block_nx))
    board_ny = math.ceil((float(n_samples)) / (2 * block_ny))
    
    # one_zero is    vs.    zero_one which is
    # 1 1 0 0               0 0 1 1
    # 1 1 0 0               0 0 1 1
    one_zero = np.concatenate((
        tf.ones((block_nx, block_ny)),
        tf.zeros((block_nx, block_ny))),
        axis=1)
    zero_one = np.flip(one_zero, axis=1)
    
    # A block is the concatenation of the former
    # 1 1 0 0
    # 1 1 0 0
    # 0 0 1 1
    # 0 0 1 1
    block = np.concatenate((one_zero, zero_one))
    
    # And a board is the tiling of many building blocks
    # 1 1 0 0  1 1 0 0 . .
    # 1 1 0 0  1 1 0 0 . .
    # 0 0 1 1  0 0 1 1 . .
    # 0 0 1 1  0 0 1 1 . .
    # . . . .  . . . .
    # . . . .  . . . .
    board = np.tile(block, ((board_nx, board_ny)))
    
    # Multiply the board with the grid element-wise, as if we were applying
    # a binary mask, but first clip the board so that it has the same shape
    # as our grid
    board = board[0:XY.shape[0], 0:XY.shape[1]]
    board = np.expand_dims(board, axis=2)
    new_grid = np.multiply(board, XY)
    sx, sy, _ = new_grid.shape
    new_grid = tf.reshape(new_grid, (sx * sy, 2))

    # Drop all rows with zero elements [0., 0.]
    # XXX: Is there any vector operation for this?
    final_grid = []
    for row in new_grid:
        if any(row):
            final_grid.append(row)
    final_grid = np.array(final_grid)

    # Add some Gaussian noise
    if add_noise:
        wx = 0.5 * (xM - xm) / n_samples    # Weighting factors
        wy = 0.5 * (yM - ym) / n_samples
        gaussian_dist = tfd.MultivariateNormalDiag(
            loc=[0., 0], scale_diag=[wx, wy])
        gaussian_noise = gaussian_dist.sample(final_grid.shape[0])
        final_grid = final_grid + gaussian_noise
    
    return final_grid


