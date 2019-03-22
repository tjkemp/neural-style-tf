import os
import time
import numpy as np
import cv2
from neuralstyle import rendering, utils
import tensorflow as tf

'''
  image loading and processing
'''


def get_init_image(init_type, content_img, style_imgs, noise_ratio, frame=None):
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        # TODO: noise_Ratio only needed for random
        init_img = get_noise_image(noise_ratio, content_img)
        return init_img

def get_content_image(content_img_dir, content_img_name, max_size):
    """ Reads and preprocesses content image. 
    
    Arguments:
    content_img_dir -- directory of content image
    content_img_name -- content image filename
    max_size -- maximum size of image
    """
    path = os.path.join(content_img_dir, content_img_name)

    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    # TODO: don't check, try to use and catch exceptions
    utils.check_image(img, path)
    img = img.astype(np.float32)
    h, w, _ = img.shape
    
    # TODO: rename variables
    mx = max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = utils.preprocess(img)
    return img


def get_style_images(content_img, style_img_dir, style_img_names):
    """ Reads and preprocesses style images. 

    Arguments:
    content_img -- content image
    style_img_dir -- directory of style image
    style_img_names -- style images' filenames

    """
    _, ch, cw, _ = content_img.shape
    style_imgs = []
    for style_fn in style_img_names:
        path = os.path.join(style_img_dir, style_fn)
        # bgr image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        utils.check_image(img, path)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = utils.preprocess(img)
        style_imgs.append(img)
    return style_imgs


def get_noise_image(noise_ratio, content_img, seed=42):
    np.random.seed(seed)
    noise_img = np.random.uniform(-20., 20.,
                                  content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1. - noise_ratio) * content_img
    return img


def get_mask_image(mask_img, width, height, content_img_dir="image_input"):
    # TODO: I can't believe I hardcoded it... content_img_dir == mask_img_dir ?
    path = os.path.join(content_img_dir, mask_img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    utils.check_image(img, path)
    img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)
    mx = np.amax(img)
    img /= mx
    return img


def convert_to_original_colors(content_img, stylized_img, color_convert_type):
    content_img = utils.postprocess(content_img)
    stylized_img = utils.postprocess(stylized_img)
    if color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = utils.preprocess(dst)
    return dst


def render_single_image(init_img_type, content_img_dir, content_img, style_img_dir, style_imgs, parameters, max_size):
    """ Main function for applying style transfer to a single image. 
    
    Arguments:
    init_img_type -- type of image, e.g. 'content', 'style', 'random'
    content_img_dir -- directory of content images
    content_img -- filename of content image
    style imgs -- ?
    maxsize -- maximum size of image ?
    """
    # TODO: implement more helpful error handling for default images and directories
    try:
        content_img = get_content_image(content_img_dir, content_img, max_size)
    except TypeError:
        print("Could not find content image {} from {}, skipping".format(content_img, content_img_dir))
        return

    try:
        style_imgs = get_style_images(content_img, style_img_dir, style_imgs)
    except TypeError:
        print("Could not find style image {} from {}, skipping".format(style_imgs[0], style_img_dir))
        return

    with tf.Graph().as_default():
        init_img = get_init_image(init_img_type, content_img, style_imgs, parameters.noise_ratio)

        tick = time.time()
        rendering.stylize(content_img, style_imgs, init_img, parameters)
        tock = time.time()

        print('Elapsed time for a single image: {}'.format(tock - tick))
