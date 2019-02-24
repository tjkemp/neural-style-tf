import os
import numpy as np
import tensorflow as tf

from neuralstyle import model, neural, utils, image

'''
  rendering -- where the magic happens
'''


def stylize(content_img, style_imgs, init_img, parameters, frame=None):
    with tf.device(parameters.device), tf.Session() as sess:
        # setup network
        net = model.build_model(content_img, parameters)

        # style loss
        if parameters.style_mask:
            L_style = neural.sum_masked_style_losses(sess, net, style_imgs, parameters)
        else:
            L_style = neural.sum_style_losses(sess, net, style_imgs, parameters)

        # content loss
        L_content = neural.sum_content_losses(sess, net, content_img, parameters)

        # denoising loss
        L_tv = tf.image.total_variation(net['input'])

        # loss weights
        alpha = parameters.content_weight
        beta = parameters.style_weight
        theta = parameters.tv_weight

        # total loss
        L_total = alpha * L_content
        L_total += beta * L_style
        L_total += theta * L_tv

        # video temporal loss
        #if parameters.video and frame > 1:
        #    gamma = parameters.temporal_weight
        #    L_temporal = loss.sum_shortterm_temporal_losses(
        #        sess, net, frame, init_img)
        #    L_total += gamma * L_temporal

        # optimization algorithm
        optimizer = get_optimizer(L_total, parameters)

        if parameters.optimizer == 'adam':
            minimize_with_adam(sess, net, optimizer, init_img, L_total, parameters.max_iterations)
        elif parameters.optimizer == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimizer, init_img)

        print("stage 1")
        output_img = sess.run(net['input'])
        print("stage 2")

        if parameters.original_colors:
            output_img = image.convert_to_original_colors(np.copy(content_img), output_img, parameters.color_convert_type)

        print("stage 3")

        if parameters.video:
            #write_video_output(frame, output_img)
            # TODO: fix this
            raise Exception("Video processing not implemented right now")
        else:
            write_image_output(output_img, content_img, style_imgs, init_img, parameters)

        print("stage 4")


def minimize_with_lbfgs(sess, net, optimizer, init_img):
    print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()

    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)


def minimize_with_adam(sess, net, optimizer, init_img, loss, max_iterations):
    print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()

    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < max_iterations):
        print("iteration {}/{} started".format(iterations, max_iterations))
        sess.run(train_op)
        print("iteration finished")
        # TODO: do something about verbose
        # TODO: args.print_iterations hard coded to 10
        #if iterations % args.print_iterations == 0 and args.verbose:
        #if iterations % 10 == 0:
        curr_loss = loss.eval()
        print("loss evaluated")
        print("At iterate {}\tf=  {}".format(iterations, curr_loss))
        iterations += 1


def get_optimizer(loss, params, verbose=True):
    """ ?
    
    Arguments:
    print_iterations -- Number of iterations between optimizer print statements
    """

    print_iterations = params.print_iterations if verbose else 0
    if params.optimizer == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B',
            options={'maxiter': params.max_iterations,
                     'disp': print_iterations})
    elif params.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
    return optimizer

def write_image_output(output_img, content_img, style_imgs, init_img, params):
    out_dir = os.path.join(params.img_output_dir, params.img_name)
    utils.maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, params.img_name + '.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    utils.write_image(img_path, output_img)
    utils.write_image(content_path, content_img)
    utils.write_image(init_path, init_img)
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
