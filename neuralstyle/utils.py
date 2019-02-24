import os
import errno
import struct
import numpy as np
import cv2

'''
  utilities and i/o
'''

def read_image(path):
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    img = img.astype(np.float32)
    img = preprocess(img)
    return img


def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)


def preprocess(img):
    """ Preprocesses an image. """
    imgpre = np.copy(img)
    # bgr to rgb
    imgpre = imgpre[..., ::-1]
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis, :, :, :]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return imgpre


def postprocess(img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    # rgb to bgr
    imgpost = imgpost[..., ::-1]
    return imgpost


def read_flow_file(path):
    with open(path, 'rb') as f:
        # 4 bytes header
        header = struct.unpack('4s', f.read(4))[0]
        # 4 bytes width, height
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        flow = np.ndarray((2, h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                flow[0, y, x] = struct.unpack('f', f.read(4))[0]
                flow[1, y, x] = struct.unpack('f', f.read(4))[0]
    return flow


def read_weights_file(path):
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(' ')))
    w = header[0]
    h = header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(' ')
        vals[i - 1] = np.array(list(map(np.float32, line)))
        vals[i - 1] = list(map(lambda x: 0. if x < 255. else 1., vals[i - 1]))
    # expand to 3 channels
    weights = np.dstack([vals.astype(np.float32)] * 3)
    return weights


def normalize(weights):
    """ Normalizes weights.

    Arguments:
    weights -- Contributions of each style layer to loss, e.g.: [0.2, 0.2, 0.2, 0.2, 0.2]
    """
    denom = sum(weights)
    if denom > 0.:
        return [float(i) / denom for i in weights]
    else:
        return [0.] * len(weights)


def maybe_make_directory(dir_path):
    """ Creates a directory if it doesn't already exist. """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def check_image(img, path):
    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)


def write_image_output(output_img, content_img, style_imgs, init_img, params):
    out_dir = os.path.join(params.img_output_dir, params.img_name)
    maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, params.img_name + '.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)
    index = 0
    for style_img in style_imgs:
        path = os.path.join(out_dir, 'style_' + str(index) + '.png')
        utils.write_image(path, style_img)
        index += 1

    # save the configuration settings
    out_file = os.path.join(out_dir, 'meta_data.txt')
    f = open(out_file, 'w')
    f.write('image_name: {}\n'.format(params.img_name))
    f.write('content: {}\n'.format(params.content_img))
    index = 0
    for style_img, weight in zip(params.style_imgs, params.style_imgs_weights):
        f.write('styles[' + str(index) +
                ']: {} * {}\n'.format(weight, style_img))
        index += 1
    index = 0
    if params.style_mask_imgs is not None:
        for mask in params.style_mask_imgs:
            f.write('style_masks[' + str(index) + ']: {}\n'.format(mask))
            index += 1
    f.write('init_type: {}\n'.format(params.init_img_type))
    f.write('content_weight: {}\n'.format(params.content_weight))
    f.write('style_weight: {}\n'.format(params.style_weight))
    f.write('tv_weight: {}\n'.format(params.tv_weight))
    f.write('content_layers: {}\n'.format(params.content_layers))
    f.write('style_layers: {}\n'.format(params.style_layers))
    f.write('optimizer_type: {}\n'.format(params.optimizer))
    f.write('max_iterations: {}\n'.format(params.max_iterations))
    f.write('max_image_size: {}\n'.format(params.max_size))
    f.close()
