from __future__ import print_function
import random
import copy
import numpy as np
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def random_flip_all_axes(x, y, prob=0.5):
    """
    Randomly flip paired 3D images (e.g., scans and labels) along one or multiple axes
    
    Input:
    - x: 3D image, optionally with an additional axis for multiple modalities. Shape: (optional channel dimension), width, height, depth
    - y: image to transform in the same way as x. Shape: same as x. 
    - prob: flip probability for each of the three iterations
    """
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([-1, -2, -3])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y


def nonlinear_transformation(x, prob=0.5, normalisation="z-score"):
    """
    Perform nonlinear intensity transformation to input image
    
    Input:
    - x: 3D image, optionally with an additional axis for multiple modalities. Shape: (optional channel dimension), width, height, depth
    - prob: transformation probability

    Returns:
    - nonlinear_x: (with probability prob) intensity transformed image of same shape as x
                   (with probability 1-prob) x
    """
    if random.random() >= prob:
        return x
    
    if normalisation == "z-score":
        x_start, x_end = np.percentile(x, 0.1), np.percentile(x, 99.9)
        points = [
            [x_start, x_start], 
            [np.random.uniform(x_start, x_end), np.random.uniform(x_start, x_end)], 
            [np.random.uniform(x_start, x_end), np.random.uniform(x_start, x_end)], 
            [x_end, x_end]
        ]
    elif normalisation == "minmax":
        points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    else:
        raise ValueError(f"Unrecognised normalisation method: {normalisation}")

    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x


def local_pixel_shuffling(x, prob=0.5):
    """
    Perform local pixel shuffling to input image, with probability prob
    
    Input:
    - x: 3D image, with an additional axis for multiple modalities. Shape: channel dimension, width, height, depth
    - prob: transformation probability per modality

    Returns:
    - local_shuffling_x: transformed image of same shape as x
    """
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    channels, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for i in range(channels):
        if random.random() >= prob:
            continue
        for _ in range(num_block):
            block_noise_size_x = random.randint(1, img_rows//10)
            block_noise_size_y = random.randint(1, img_cols//10)
            block_noise_size_z = random.randint(1, img_deps//10)
            noise_x = random.randint(0, img_rows-block_noise_size_x)
            noise_y = random.randint(0, img_cols-block_noise_size_y)
            noise_z = random.randint(0, img_deps-block_noise_size_z)
            window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                                noise_y:noise_y+block_noise_size_y, 
                                noise_z:noise_z+block_noise_size_z,
                            ]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x, 
                                    block_noise_size_y, 
                                    block_noise_size_z))
            image_temp[i, noise_x:noise_x+block_noise_size_x, 
                        noise_y:noise_y+block_noise_size_y, 
                        noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x


def image_in_painting(x):
    """
    Perform image inpainting to input image
    
    Input:
    - x: 3D image. Shape: width, height, depth

    Returns:
    - x: transformed image of same shape as x
    """
    img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x


def image_out_painting(x):
    """
    Perform image outpainting to input image
    
    Input:
    - x: 3D image. Shape: width, height, depth

    Returns:
    - x: transformed image of same shape as x
    """
    img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(*x.shape) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x
                

def generate_single_pair(y, config):
    """
    Generate data augmentations to a single image with probabilities specified in the config
    
    Input:
    - y: original 3D image, with an additional axis for multiple modalities. Shape: channel dimension, width, height, depth.

    Returns:
    - x: transformed 3D image. Shape: channel dimension, width, height, depth.
    - y: original 3D image. Shape: channel dimension, width, height, depth.
    """
    # Autoencoder
    x = copy.deepcopy(y)
    
    # Flip
    x, y = random_flip_all_axes(x, y, config.flip_rate)

    # Local Shuffle Pixel
    x = local_pixel_shuffling(x, prob=config.local_rate)
    
    # Apply non-Linear transformation with an assigned probability
    x = nonlinear_transformation(x, config.nonlinear_rate)
    
    # Inpainting & Outpainting
    channels = x.shape[0]
    for i in range(channels):
        if random.random() < config.paint_rate:
            if random.random() < config.inpaint_rate:
                # Inpainting
                x[i] = image_in_painting(x[i])
            else:
                # Outpainting
                x[i] = image_out_painting(x[i])
    
    return x, y


def generate_pair(img, batch_size, config, status="test"):
    """
    img: Images, shape (bs; channel; z; y; x)
    """
    img_rows, img_cols, img_deps = img.shape[2], img.shape[3], img.shape[4]
    while True:
        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:batch_size]]
        x = copy.deepcopy(y)
        for n in range(batch_size):
            # apply augmentations
            x[n], y[n] = generate_single_pair(y[n], config=config)
        
        yield (x, y)
