import os
import time
import struct
import argparse
import itertools

from neuralstyle import utils, image

'''
  parsing and configuration
'''


def parse_args():

    desc = "TensorFlow implementation of 'A Neural Algorithm for Artistic Style'"
    parser = argparse.ArgumentParser(description=desc)

    # options for single image
    parser.add_argument('--verbose', action='store_true',
                        help='Boolean flag indicating if statements should be printed to the console.')

    parser.add_argument('--img_name', type=str,
                        default='result',
                        help='Filename of the output image.')

    parser.add_argument('--batch', action='store_true',
                        help='Stylize all images in input directory with all images in style directory.')

    parser.add_argument('--style_imgs', nargs='+', type=str,
                        help='Filenames of the style images (example: starry-night.jpg)')

    parser.add_argument('--style_imgs_weights', nargs='+', type=float,
                        default=[1.0],
                        help='Interpolation weights of each of the style images. (example: 0.5 0.5)')

    parser.add_argument('--content_img', type=str,
                        help='Filename of the content image (example: lion.jpg)')

    parser.add_argument('--style_imgs_dir', type=str,
                        default='styles',
                        help='Directory path to the style images. (default: %(default)s)')

    parser.add_argument('--content_img_dir', type=str,
                        default='image_input',
                        help='Directory path to the content image. (default: %(default)s)')

    parser.add_argument('--init_img_type', type=str,
                        default='content',
                        choices=['random', 'content', 'style'],
                        help='Image used to initialize the network. (default: %(default)s)')

    parser.add_argument('--max_size', type=int,
                        default=512,
                        help='Maximum width or height of the input images. (default: %(default)s)')

    parser.add_argument('--content_weight', type=float,
                        default=5e0,
                        help='Weight for the content loss function. (default: %(default)s)')

    parser.add_argument('--style_weight', type=float,
                        default=1e4,
                        help='Weight for the style loss function. (default: %(default)s)')

    parser.add_argument('--tv_weight', type=float,
                        default=1e-3,
                        help='Weight for the total variational loss function. Set small (e.g. 1e-3). (default: %(default)s)')

    parser.add_argument('--temporal_weight', type=float,
                        default=2e2,
                        help='Weight for the temporal loss function. (default: %(default)s)')

    parser.add_argument('--content_loss_function', type=int,
                        default=1,
                        choices=[1, 2, 3],
                        help='Different constants for the content layer loss function. (default: %(default)s)')

    parser.add_argument('--content_layers', nargs='+', type=str,
                        default=['conv4_2'],
                        help='VGG19 layers used for the content image. (default: %(default)s)')

    parser.add_argument('--style_layers', nargs='+', type=str,
                        default=[
                            'relu1_1',
                            'relu2_1',
                            'relu3_1',
                            'relu4_1',
                            'relu5_1'],
                        help='VGG19 layers used for the style image. (default: %(default)s)')

    parser.add_argument('--content_layer_weights', nargs='+', type=float,
                        default=[1.0],
                        help='Contributions (weights) of each content layer to loss. (default: %(default)s)')

    parser.add_argument('--style_layer_weights', nargs='+', type=float,
                        default=[0.2, 0.2, 0.2, 0.2, 0.2],
                        help='Contributions (weights) of each style layer to loss. (default: %(default)s)')

    parser.add_argument('--original_colors', action='store_true',
                        help='Transfer the style but not the colors.')

    parser.add_argument('--color_convert_type', type=str,
                        default='yuv',
                        choices=['yuv', 'ycrcb', 'luv', 'lab'],
                        help='Color space for conversion to original colors (default: %(default)s)')

    parser.add_argument('--color_convert_time', type=str,
                        default='after',
                        choices=['after', 'before'],
                        help='Time (before or after) to convert to original colors (default: %(default)s)')

    parser.add_argument('--style_mask', action='store_true',
                        help='Transfer the style to masked regions.')

    parser.add_argument('--style_mask_imgs', nargs='+', type=str,
                        default=None,
                        help='Filenames of the style mask images (example: face_mask.png) (default: %(default)s)')

    parser.add_argument('--noise_ratio', type=float,
                        default=1.0,
                        help="Interpolation value between the content image and noise image if the network is initialized with 'random'.")

    parser.add_argument('--seed', type=int,
                        default=0,
                        help='Seed for the random number generator. (default: %(default)s)')

    parser.add_argument('--model_weights', type=str,
                        default='imagenet-vgg-verydeep-19.mat',
                        help='Weights and biases of the VGG-19 network.')

    parser.add_argument('--pooling_type', type=str,
                        default='avg',
                        choices=['avg', 'max'],
                        help='Type of pooling in convolutional neural network. (default: %(default)s)')

    parser.add_argument('--device', type=str,
                        default='/gpu:0',
                        choices=['/gpu:0', '/cpu:0'],
                        help='GPU or CPU mode.  GPU mode requires NVIDIA CUDA. (default|recommended: %(default)s)')

    parser.add_argument('--img_output_dir', type=str,
                        default='image_output',
                        help='Relative or absolute directory path to output image and data.')

    # optimizations
    parser.add_argument('--optimizer', type=str,
                        default='lbfgs',
                        choices=['lbfgs', 'adam'],
                        help='Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. (default|recommended: %(default)s)')

    parser.add_argument('--learning_rate', type=float,
                        default=1e0,
                        help='Learning rate parameter for the Adam optimizer. (default: %(default)s)')

    parser.add_argument('--max_iterations', type=int,
                        default=1000,
                        help='Max number of iterations for the Adam or L-BFGS optimizer. (default: %(default)s)')

    parser.add_argument('--print_iterations', type=int,
                        default=50,
                        help='Number of iterations between optimizer print statements. (default: %(default)s)')

    args = parser.parse_args()

    # normalize weights
    args.style_layer_weights = utils.normalize(args.style_layer_weights)
    args.content_layer_weights = utils.normalize(args.content_layer_weights)
    args.style_imgs_weights = utils.normalize(args.style_imgs_weights)

    # create directories for output
    utils.maybe_make_directory(args.img_output_dir)

    return args


def main():
    args = parse_args()

    if args.batch:
        content_images = os.listdir(args.content_img_dir)
        style_images = os.listdir(args.style_imgs_dir)
        images_to_render = [(pair[0], pair[1]) for pair in itertools.product(content_images, style_images)]

        for content_img, style_img in images_to_render:
            image.render_single_image(
                args.init_img_type,
                args.content_img_dir,
                content_img,
                args.style_imgs_dir,
                style_img,
                args,
                args.max_size)

    else:
        image.render_single_image(
            args.init_img_type,
            args.content_img_dir,
            args.content_img,
            args.style_imgs_dir,
            args.style_imgs,
            args,
            args.max_size)


if __name__ == '__main__':
    main()
