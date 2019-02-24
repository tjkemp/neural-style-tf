def get_content_frame(frame):
    fn = args.content_frame_frmt.format(str(frame).zfill(4))
    path = os.path.join(args.video_input_dir, fn)
    img = read_image(path)
    return img


def get_prev_frame(frame):
    # previously stylized frame
    prev_frame = frame - 1
    fn = args.content_frame_frmt.format(str(prev_frame).zfill(4))
    path = os.path.join(args.video_output_dir, fn)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    check_image(img, path)
    return img


def get_prev_warped_frame(frame):
    prev_img = get_prev_frame(frame)
    prev_frame = frame - 1
    # backwards flow: current frame -> previous frame
    fn = args.backward_optical_flow_frmt.format(str(frame), str(prev_frame))
    path = os.path.join(args.video_input_dir, fn)
    flow = read_flow_file(path)
    warped_img = warp_image(prev_img, flow).astype(np.float32)
    img = preprocess(warped_img)
    return img


def get_content_weights(frame, prev_frame):
    forward_fn = args.content_weights_frmt.format(str(prev_frame), str(frame))
    backward_fn = args.content_weights_frmt.format(str(frame), str(prev_frame))
    forward_path = os.path.join(args.video_input_dir, forward_fn)
    backward_path = os.path.join(args.video_input_dir, backward_fn)
    forward_weights = read_weights_file(forward_path)
    backward_weights = read_weights_file(backward_path)
    return forward_weights  # , backward_weights


def warp_image(src, flow):
    _, h, w = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[1, y, :] = float(y) + flow[1, y, :]
    for x in range(w):
        flow_map[0, :, x] = float(x) + flow[0, :, x]
    # remap pixels to optical flow
    dst = cv2.remap(
        src, flow_map[0], flow_map[1],
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def render_video():
    for frame in range(args.start_frame, args.end_frame + 1):
        with tf.Graph().as_default():
            print(
                '\n---- RENDERING VIDEO FRAME: {}/{} ----\n'.format(frame, args.end_frame))
            if frame == 1:
                content_frame = get_content_frame(frame)
                style_imgs = get_style_images(content_frame)
                init_img = get_init_image(
                    args.first_frame_type, content_frame, style_imgs, frame)
                args.max_iterations = args.first_frame_iterations
                tick = time.time()
                stylize(content_frame, style_imgs, init_img, frame)
                tock = time.time()
                print('Frame {} elapsed time: {}'.format(frame, tock - tick))
            else:
                content_frame = get_content_frame(frame)
                style_imgs = get_style_images(content_frame)
                init_img = get_init_image(
                    args.init_frame_type, content_frame, style_imgs, frame)
                args.max_iterations = args.frame_iterations
                tick = time.time()
                stylize(content_frame, style_imgs, init_img, frame)
                tock = time.time()
                print('Frame {} elapsed time: {}'.format(frame, tock - tick))

def write_video_output(frame, output_img):
    fn = args.content_frame_frmt.format(str(frame).zfill(4))
    path = os.path.join(args.video_output_dir, fn)
    utils.write_image(path, output_img)
