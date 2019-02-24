import tensorflow as tf

#TODO: resolve the location of get_mask_image
from neuralstyle import image
'''
  'a neural algorithm for artistic style' loss functions
'''


def content_layer_loss(p, x, content_loss_function):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if content_loss_function == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
    elif content_loss_function == 2:
        K = 1. / (N * M)
    elif content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1. / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G


def mask_style_layer(a, x, mask_img):
    _, h, w, d = a.get_shape()
    mask = image.get_mask_image(mask_img, w.value, h.value)
    mask = tf.convert_to_tensor(mask)
    tensors = []
    for _ in range(d.value):
        tensors.append(mask)
    mask = tf.stack(tensors, axis=2)
    mask = tf.stack(mask, axis=0)
    mask = tf.expand_dims(mask, 0)
    a = tf.multiply(a, mask)
    x = tf.multiply(x, mask)
    return a, x


def sum_masked_style_losses(sess, net, style_imgs, params):
    """ ? """
    total_style_loss = 0.
    weights = params.style_imgs_weights
    masks = params.style_mask_imgs
    for img, img_weight, img_mask in zip(style_imgs, weights, masks):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(params.style_layers, params.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            a, x = mask_style_layer(a, x, img_mask)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(params.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss


def sum_style_losses(sess, net, style_imgs, params):
    """ ? """
    total_style_loss = 0.
    weights = params.style_imgs_weights
    for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(params.style_layers, params.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(params.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss


def sum_content_losses(sess, net, content_img, params):
    """ ? """
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(params.content_layers, params.content_layer_weights):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(p)
        content_loss += content_layer_loss(p, x, params.content_loss_function) * weight
    content_loss /= float(len(params.content_layers))
    return content_loss
