import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import myutils


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
        center_coord = np.array([x_center, y_center])
    else:
        x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
        y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
        center_coord = np.array([x_center, y_center]).reshape((bboxes.shape[0], 2))

    return center_coord

def get_center_grid_of_bbox(img, label, S):
    """
    find the index of the grid that the center of bounding box locates in.

    :param img:
    :param label:
    :param S:
    :return: (row_idx, col_idx), the idx starts from 0.
    """
    x_center, y_center = get_center_coord_of_bbox(label[0, 1:])
    height, width = img.shape[:2]
    x_interval = width / S
    y_interval = height / S
    for i in range(S):
        if x_interval * i < x_center < x_interval * (i+1):
            col_idx = i
        if y_interval * i < y_center < y_interval * (i+1):
            row_idx = i
    return row_idx, col_idx


def translate_label(img, label, S=7):
    """
    translate the label data from original format to yolo format

    :param img: np.array, int8, (h, w, c)
    :param label: np.array, float32, (N, 5)
    :param S: the number of grids on one axis
    :return: the translated yolo label. np.array, float32, (N, 5), (cls_id, ct_x, ct_y, rel_w, rel_h)
    """
    center_row, center_col = get_center_grid_of_bbox(img, label, S)

    height, width = img.shape[:2]
    row_interval = height / S
    col_interval = width / S
    grid_origin = np.array([center_col * col_interval, center_row * row_interval])

    center_coord = get_center_coord_of_bbox(label[:, 1:])
    center_coord = center_coord.flatten()
    rel_center = center_coord - grid_origin
    rel_center = rel_center / [col_interval, row_interval]

    abs_right_bottom_corner = label[0, -2:]
    half_w_h = abs_right_bottom_corner - [center_coord[0], center_coord[1]]
    full_w_h = half_w_h * 2
    rel_w_h = full_w_h / np.array([width, height])

    yolo_label = np.concatenate((rel_center, rel_w_h), axis=-1)
    yolo_label = np.concatenate((label[0, 0:1], yolo_label), axis=-1)
    return yolo_label


def generate_target(img, label, tensor_pred):
    """
    generate the target of one image for the loss calculation.

    :param img: np.array, uint8, (h, w, c)
    :param label: np.array, int32, absolute, shape: (N, 5), [class_id, x1, y1, x2, y2]
    :param pred_tensor: np.array, float32, (S, S, (B*5 + C))
    :return: target tensor, np.array, float32, (S, S, (B*5 + C))
    """

    height, width = img.shape[:2]

    S = int(tensor_pred.shape[0])
    B = int((tensor_pred.shape[-1] - 20) // 5)

    grid_row, grid_col = get_center_grid_of_bbox(img, label, S)
    tensor_targ = np.zeros(tensor_pred.shape)

    # box confidence
    for i in range(B):
        if i == 0:
            boxes_pred = tensor_pred[grid_row, grid_col, i*5:i*5+4]
            boxes_pred = np.expand_dims(boxes_pred, axis=0)
            continue
        temp_boxes_pred = tensor_pred[grid_row, grid_col, i*5:i*5+4]
        temp_boxes_pred = np.expand_dims(temp_boxes_pred, axis=0)
        boxes_pred = np.concatenate((boxes_pred, temp_boxes_pred), axis=0)
    print('boxes pred:', boxes_pred)

    rltv_label_box = myutils.bbox_abs_to_rel(label[0, 1:], img.shape[:2])
    iou = mx.nd.contrib.box_iou(mx.nd.array(boxes_pred), mx.nd.array(rltv_label_box.reshape((1, 4))))
    print('iou:', iou)

    idx_max_iou = np.argmax(iou.asnumpy())
    print('idx_max_iou:', idx_max_iou)

    tensor_targ[grid_row, grid_col, idx_max_iou*5+4] = 1
    print(tensor_targ[grid_row, grid_col])

    # box coordinates
    yolo_label = translate_label(img, label, S)
    print('yolo label:', yolo_label)

    tensor_targ[grid_row, grid_col, idx_max_iou*5:idx_max_iou*5+4] = yolo_label[1:]

    # class probability
    tensor_targ[grid_row, grid_col, int(B*5 + yolo_label[0])] = 1
    print('tensor_targ:')
    print(tensor_targ[grid_row, grid_col])

    # visualize
    fig = plt.imshow(img)
    axes = fig.axes
    myutils._add_rectangle(axes, rltv_label_box)

    for box in boxes_pred:
        myutils._add_rectangle(axes, box, color='blue')
    plt.show()

    grid_mask = np.zeros((7, 7, 30))
    grid_mask[grid_row, grid_col, :] = 1

    return tensor_targ


def calc_yolo_loss(tensor_pred, tensor_targ, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
    """
    calculate the loss function of yolo algorithm.

    :param tensor_pred: np.array, (S, S, (B*5 + C))
    :param tensor_targ: np.array, (S, S, (B*5 + C))
    :return:
    """
    lambda_coord = 5
    lambda_noobj = 0.5

    box_pred = tensor_pred[:, :, :B*5]
    box_targ = tensor_targ[:, :, :B*5]
    cls_pred = tensor_pred[:, :, B*5:]
    cls_targ = tensor_targ[:, :, B*5:]

    mask_obj = np.zeros(tensor_targ.shape[:-1])
    mask_obj[tensor_targ[:, :, 4] == 1] = 1

    x_pred, y_pred = box_pred[:, :, 0], box_pred[:, :, 1]
    x_targ, y_targ = box_targ[:, :, 0], box_targ[:, :, 1]
    loss = lambda_coord * np.sum(((x_pred - x_targ)**2 + (y_pred - y_targ)**2) * mask_obj, axis=(0, 1))

    w_pred, h_pred = box_pred[:, :, 2], box_pred[:, :, 3]
    w_targ, h_targ = box_targ[:, :, 2], box_targ[:, :, 3]
    loss += lambda_coord * np.sum(((np.sqrt(w_pred) - np.sqrt(w_targ))**2 +
                                   (np.sqrt(h_pred) - np.sqrt(h_targ))**2) * mask_obj, axis=(0, 1))

    c_pred = box_pred[:, :, 4]
    c_targ = box_targ[:, :, 4]
    loss += np.sum((c_pred - c_targ)**2 * mask_obj, axis=(0, 1))

    mask_noobj = np.zeros(mask_obj.shape)
    mask_noobj[mask_obj == 0] = 1
    loss += lambda_noobj * np.sum((c_pred - c_targ)**2 * mask_noobj, axis=(0, 1))

    temp_value = np.sum((cls_pred - cls_targ)**2, axis=-1) * mask_obj
    loss += np.sum(temp_value, axis=(0, 1))

    print('loss:', loss)
    return loss


