def get_longterm_weights(i, j):
    c_sum = 0.
    for k in range(args.prev_frame_indices):
        if i - k > i - j:
            c_sum += get_content_weights(i, i - k)
    c = get_content_weights(i, i - j)
    c_max = tf.maximum(c - c_sum, 0.)
    return c_max


def sum_longterm_temporal_losses(sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    loss = 0.
    for j in range(args.prev_frame_indices):
        prev_frame = frame - j
        w = get_prev_warped_frame(frame)
        c = get_longterm_weights(frame, prev_frame)
        loss += temporal_loss(x, w, c)
    return loss


def sum_shortterm_temporal_losses(sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    prev_frame = frame - 1
    w = get_prev_warped_frame(frame)
    c = get_content_weights(frame, prev_frame)
    loss = temporal_loss(x, w, c)
    return loss
