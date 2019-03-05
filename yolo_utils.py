import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import myutils

transformer = mx.gluon.data.vision.transforms.Compose(
    [mx.gluon.data.vision.transforms.Resize(size=(448, 448)),
    mx.gluon.data.vision.transforms.ToTensor(),
    mx.gluon.data.vision.transforms.Normalize(mean=myutils.mean, std=myutils.std)])


def transform_image(img):
    trsfm_img = transformer(mx.nd.array(img))
    trsfm_img = mx.nd.expand_dims(trsfm_img, axis=0)
    return trsfm_img


def visualize_grids(img, label, S=7):
    """
    plot grids and label bounding boxes on the given image.

    :param img: np.array, uint8, (h, w, c)
    :param label: np.array, int32, (N, 5), N represents the id of bbox, the 5 represents (cls_id, x1, y1, x2, y2)
    :param S: the image is divided by S * S grids
    :return: None
    """

    fig = plt.imshow(img)
    axes = fig.axes

    height, width = img.shape[:2]
    x_interval = width / S
    y_interval = height / S

    grid_line_start_point = []
    grid_line_end_point = []
    for i in range(S+1):
        grid_line_start_point.append([x_interval * i, 0])
        grid_line_end_point.append([x_interval * i, height])
        grid_line_start_point.append([0, y_interval * i])
        grid_line_end_point.append([width, y_interval * i])

    for i in range(len(grid_line_start_point)):
        x_coords, y_coords = zip(*(grid_line_start_point[i], grid_line_end_point[i]))
        plt.plot(x_coords, y_coords, 'b-', linewidth=1)

    axes.set_xmargin(0)
    axes.set_ymargin(0)

    rltv_bbox = myutils.bbox_abs_to_rel(bbox=label[0, 1:], pic_size=img.shape[:2])
    myutils._add_rectangle(axes, rltv_bbox)

    x_center, y_center = get_center_coord_of_bbox(label[0, 1:])
    plt.plot(x_center, y_center, 'r.', markersize=15)

    return fig


def get_center_coord_of_bbox(bboxes):
    """
    calculate the center point of bounding boxes.

    :param bboxes: np.array, int32, absolute, (N, 5) or (4, )
    :return: np.array, float64, absolute, (N, 2) or (2, ), as x and y
    """

    if len(bboxes.shape) == 1:
        x_center = (bboxes[0] + bboxes[2]) / 2
        y_center = (bboxes[1] + bboxes[3]) / 2
        center_coord = np.array([x_center, y_center]).flatten()
    else:
        x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
        y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
        center_coord = np.array([x_center, y_center]).reshape((bboxes.shape[0], 2))

    return center_coord


def get_center_grid_of_bbox(img, box, S):
    """
    find the index of the grid that the center of bounding box locates in.

    :param img:
    :param box:
    :param S:
    :return: (row_idx, col_idx), the idx starts from 0.
    """

    x_center, y_center = get_center_coord_of_bbox(box)
    height, width = img.shape[:2]
    x_interval = width / S
    y_interval = height / S
    for i in range(S):
        if x_interval * i < x_center < x_interval * (i+1):
            col_idx = i
        if y_interval * i < y_center < y_interval * (i+1):
            row_idx = i
    return row_idx, col_idx


def translate_box_yolo_to_abs(img, box_yolo, grid_row, grid_col, S=7):
    box_yolo = box_yolo.copy()

    height, width = img.shape[:2]
    row_interval = height / S
    col_interval = width / S

    coor_center = box_yolo[:2] * (col_interval, row_interval) + \
                   (col_interval * grid_col, row_interval * grid_row)
    width_height = box_yolo[2:4] * (width, height)

    top_left_corner = coor_center - width_height / 2
    bottom_right_corner = coor_center + width_height / 2

    return np.array((top_left_corner, bottom_right_corner)).flatten()


def translate_box_abs_to_yolo(img, box_abs, S=7):
    box_abs = box_abs.copy()

    center_row, center_col = get_center_grid_of_bbox(img, box_abs, S)

    height, width = img.shape[:2]
    row_interval = height / S
    col_interval = width / S
    grid_origin = np.array([center_col * col_interval, center_row * row_interval])

    center_coord = get_center_coord_of_bbox(box_abs)
    center_coord = center_coord.flatten()
    yolo_center = (center_coord - grid_origin) / [col_interval, row_interval]

    half_w_h = box_abs[2:4] - center_coord
    full_w_h = half_w_h * 2
    rel_w_h = full_w_h / [width, height]

    return np.array([yolo_center, rel_w_h]).flatten()


def translate_label(img, label, S=7):
    """
    translate the label data from original format to yolo format

    :param img: np.array, int8, (h, w, c)
    :param label: np.array, float32, (N, 5)
    :param S: the number of grids on one axis
    :return: the translated yolo label. np.array, float32, (N, 5), (cls_id, ct_x, ct_y, rel_w, rel_h)
    """

    yolo_label = translate_box_abs_to_yolo(img, label[0, 1:])
    yolo_label = np.concatenate((label[0, 0:1], yolo_label), axis=-1)
    return yolo_label


def _generate_random_pred_tensor():
    def generate_random_box():
        while True:
            box = np.random.uniform(0, 1, size=(4, ))
            if box[0] < box[2] and box[1] < box[3]:
                break
        return box.reshape((1, 4))

    boxes = np.empty(shape=(2, 4))
    for i in range(2):
        boxes[i] = generate_random_box()
    print(boxes)
    pred_tensor = np.random.uniform(size=(7, 7, 30))
    pred_tensor[4, 2, 0:4] = boxes[0]
    pred_tensor[4, 2, 5:9] = boxes[1]
    return pred_tensor


def generate_target(img, label, tensor_pred):
    """
    generate the target of one image for the loss calculation.

    :param img: np.array, uint8, (h, w, c)
    :param label: np.array, int32, absolute, shape: (N, 5), [class_id, x1, y1, x2, y2]
    :param tensor_pred: np.array, float32, (S, S, (B*5 + C))
    :return: target tensor, np.array, float32, (S, S, (B*5 + C))
    """

    tensor_pred = tensor_pred.copy()
    tensor_targ = np.zeros(tensor_pred.shape)

    height, width = img.shape[:2]
    S = int(tensor_pred.shape[0])
    B = int((tensor_pred.shape[-1] - 20) // 5)
    grid_row, grid_col = get_center_grid_of_bbox(img, label[0, 1:], S)

    print('tensor predicted:')
    print(tensor_pred[grid_row, grid_col])

    # box confidence
    for i in range(B):
        if i == 0:
            boxes_pred = tensor_pred[grid_row, grid_col, i*5:i*5+4]
            boxes_pred = np.expand_dims(boxes_pred, axis=0)
            continue
        temp_boxes_pred = tensor_pred[grid_row, grid_col, i*5:i*5+4]
        temp_boxes_pred = np.expand_dims(temp_boxes_pred, axis=0)
        boxes_pred = np.concatenate((boxes_pred, temp_boxes_pred), axis=0)

    boxes_rel = np.empty(boxes_pred.shape)
    for i, box in enumerate(boxes_pred):
        box_abs = translate_box_yolo_to_abs(img, box, grid_row, grid_col)
        box_rel = myutils.bbox_abs_to_rel(box_abs, (height, width))
        boxes_rel[i] = box_rel

    rltv_label_box = myutils.bbox_abs_to_rel(label[0, 1:], img.shape[:2])
    iou = mx.nd.contrib.box_iou(mx.nd.array(boxes_rel), mx.nd.array(rltv_label_box.reshape((1, 4))))

    idx_max_iou = np.argmax(iou.asnumpy())

    tensor_targ[grid_row, grid_col, idx_max_iou*5+4] = 1

    # box coordinates
    yolo_label = translate_label(img, label, S)
    tensor_targ[grid_row, grid_col, idx_max_iou*5:idx_max_iou*5+4] = yolo_label[1:]

    # class probability
    tensor_targ[grid_row, grid_col, int(B*5 + yolo_label[0])] = 1

    print('tensor target:')
    print(tensor_targ[grid_row, grid_col])

    # visualize
    fig = plt.imshow(img)
    axes = fig.axes
    myutils._add_rectangle(axes, rltv_label_box)

    for box in boxes_rel:
        myutils._add_rectangle(axes, box, color='blue')
    plt.show()

    grid_mask = np.zeros((7, 7, 30))
    grid_mask[grid_row, grid_col, :] = 1

    return tensor_targ


def calc_yolo_loss(tensor_pred, tensor_targ, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
    """
    calculate the loss function of yolo algorithm.

    :param tensor_pred: mx.nd.array, (S, S, (B*5 + C))
    :param tensor_targ: np.array, (S, S, (B*5 + C))
    :return:
    """
    tensor_targ = mx.nd.array(tensor_targ)

    lambda_coord = 5
    lambda_noobj = 0.5

    box_pred = tensor_pred[:, :, :B*5]
    box_targ = tensor_targ[:, :, :B*5]
    cls_pred = tensor_pred[:, :, B*5:]
    cls_targ = tensor_targ[:, :, B*5:]

    mask_obj = np.zeros(tensor_targ.shape[:-1])
    mask_obj[tensor_targ[:, :, 4].asnumpy() == 1] = 1
    mask_obj = mx.nd.array(mask_obj)

    x_pred, y_pred = box_pred[:, :, 0], box_pred[:, :, 1]
    x_targ, y_targ = box_targ[:, :, 0], box_targ[:, :, 1]
    loss = lambda_coord * mx.nd.sum(((x_pred - x_targ)**2 + (y_pred - y_targ)**2) * mask_obj, axis=(0, 1))

    w_pred, h_pred = box_pred[:, :, 2], box_pred[:, :, 3]
    w_targ, h_targ = box_targ[:, :, 2], box_targ[:, :, 3]
    loss = loss + lambda_coord * mx.nd.sum(((mx.nd.sqrt(w_pred) - mx.nd.sqrt(w_targ))**2 +
                  (mx.nd.sqrt(h_pred) - mx.nd.sqrt(h_targ))**2) * mask_obj, axis=(0, 1))

    c_pred = box_pred[:, :, 4]
    c_targ = box_targ[:, :, 4]
    loss = loss + mx.nd.sum((c_pred - c_targ)**2 * mask_obj, axis=(0, 1))

    mask_noobj = np.zeros(mask_obj.shape)
    mask_noobj[mask_obj.asnumpy() == 0] = 1
    mask_noobj = mx.nd.array(mask_noobj)
    loss = loss + lambda_noobj * mx.nd.sum((c_pred - c_targ)**2 * mask_noobj, axis=(0, 1))

    temp_value = mx.nd.sum((cls_pred - cls_targ)**2, axis=-1) * mask_obj
    loss = loss + mx.nd.sum(temp_value, axis=(0, 1))

    print('loss:', loss)
    return loss
