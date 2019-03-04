import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# data processing

def resize_img_and_label (img, label, size=300):
    """
    :function: resize the img and label to (300, 300), and transform the dtype from int8 to float32
    :param img: np.array, int8, (h, w, c)
    :param label: absolute, np.array, (1, 5)
    :param size: tuple (height, width)
    :return: (resz_img, resz_label),  np.array, dtype of resz_img: float32, absolute coordinates
    """
    if type(size) == 'int':
        keep_ratio = True
    else:
        keep_ratio = False

    import copy
    bbox = label[0:1, 1:] / ([img.shape[1], img.shape[0]] * 2)
    bbox = bbox * 300
    resz_label = copy.deepcopy(label)
    resz_label[0:1, 1:] = bbox

    '''
    关于为什么要在这里变成float32的说明：
    因为插值必定会引入小数，所以原数据经过插值就会从int8类型变成float32类型，并且范围在0到255
    这样的一个数据放到 matplotlib 里面画图会有警告，说数据类型或者数值范围不对
    所以我要在这里把它在数值上归一化到0~1，保证出来的数据能直接用于画图
    这样就会带来一个问题，
    因为在 mxnet.gluon.data.vision.transforms.ToTensor() 里，不光变换维度，还会将 int 类型的数据转换成 float 型，
    这样就与我的操作产生了矛盾，因此用完我这个 resize 函数，就不要再用 ToTensor() 了，
    应该改成我自己写的 to_tensor() 函数（仅用于改变维度，改变数据类型）
    '''
    resz_img = img / 255
    resz_img = mx.nd.array(resz_img)
    resize = mx.gluon.data.vision.transforms.Resize(size=size, keep_ratio=keep_ratio)
    resz_img = resize(resz_img)

    return resz_img.asnumpy(), resz_label


def to_tensor(img):
    """
    :function: change the dimensions and the type of data (from np.array to mx.nd.array)
    :param img: np.array, (h, w, c)
    :return: mx.nd.array, (c, h, w)
    """
    return mx.nd.array(img).transpose(axes=(2, 0, 1))


def prepare_datum_for_net(img, label):
    """
    :func: image transform includes resizing, color normalizing, to tensor, expanding dimension
    label transform includes converting to relative coords, expanding dimension

    :param img: np.array, uint8, (h, w, c)
    :param label: np.array, uint8, (N, 5)
    :return:
    """
    mx_img = img.astype('float32')  # deepcopy
    mx_label = label.astype('float32')

    mx_img, _ = resize_img_and_label(mx_img, mx_label, size=416)
    mx_img = mx.img.color_normalize(mx.nd.array(mx_img), mean=mx.nd.array(mean), std=mx.nd.array(std))
    mx_img = to_tensor(mx_img)
    mx_img = mx_img.expand_dims(axis=0)

    mx_label[:, 1:] = bbox_abs_to_rel(mx_label[:, 1:], img.shape[:2])
    mx_label = mx.nd.array(mx_label)
    mx_label = mx_label.expand_dims(axis=0)

    return mx_img, mx_label

# data visualization

def _add_rectangle(axes, relative_bbox, color='red'):
    """
    :param axes:
    :param relative_bbox: relative_bbox: numpy.ndarray, (x_left_top, y_left_top, x_right_bottom, y_right_bottom)
    :param color: color: 'red', 'blue', '...'
    :return: None
    """

    img_lst = axes.get_images()
    img = img_lst[0]
    pic_size = img.get_size()
    h, w = pic_size
    dx1, dy1 = relative_bbox[0].tolist(), relative_bbox[1].tolist()
    dx2, dy2 = relative_bbox[2].tolist(), relative_bbox[3].tolist()
    x1, y1 = dx1 * w, dy1 * h
    x2, y2 = dx2 * w, dy2 * h
    axes.add_patch(
        plt.Rectangle(xy=(x1, y1), width=x2 - x1, height=y2 - y1, fill=False, edgecolor=color, linewidth=2))
    return


def data_visualize(img, bboxes):
    """
    :param img: numpy.ndarray, (h, w, c)
    :param bboxes: absolute position, numpy.ndarray, (N, 4), (x_left_top, y_left_top, x_right_bottom, y_right_bottom)
    :return: the figure that pyplot uses.
    """
    fig = plt.imshow(img)
    axes = fig.axes

    for bbox in bboxes:
        rel_bbox = bbox_abs_to_rel(bbox, img.shape[:2])
        _add_rectangle(axes, rel_bbox)

    plt.show()
    return fig


def target_visualize(img, label, anchors, cls_preds):
    height, width = img.shape[0:2]
    rel_label = label[0, 1:]
    rel_label = rel_label.astype('float32') / ([width, height] * 2)

    mx_label = rel_label.reshape((1, 1, 4))  # transform the shape to mxnet format (b, 1, 5), float32
    class_digit = np.array([1.]).reshape((1, 1, 1))
    mx_label = np.concatenate((class_digit, mx_label), axis=2)

    box_target, box_mask, cls_target = mx.nd.contrib.MultiBoxTarget(anchor=mx.nd.array(anchors),
                                                                    cls_pred=mx.nd.array(cls_preds).transpose(
                                                                        (0, 2, 1)),
                                                                    label=mx.nd.array(mx_label),
                                                                    overlap_threshold=0.25)
    idx = cls_target.asnumpy().astype('int8').flatten()
    targ = anchors.asnumpy()[0, idx > 0.9]  # idx == 1

    figure = plt.imshow(img)
    axes = figure.axes

    for box in targ:
        _add_rectangle(axes=axes, relative_bbox=box, color='blue')

    _add_rectangle(axes=axes, relative_bbox=rel_label, color='red')

    plt.show()
    return


def anchor_visualize(img, anchors, i):
    """
    :function: visualize the i-th group of anchors. if i is greater than bounding, this function will give warning.
    :param img: numpy.array, float32 or int8, (h, w, c)
    :param anchors: np.array, (1, N, 4)
    :param i: the i-th group
    :return: None
    """
    total_num = anchors.shape[1]
    if i*5+5 > total_num:
        print('There is no %i-th group of anchors, and the maximum of i is' % i, total_num//5-1)
        return
    bboxes = anchors[0, i*5:i*5+5, :]
    bboxes = bbox_rel_to_abs(bbox=bboxes.asnumpy(), pic_size=img.shape[:2])
    data_visualize(img, bboxes)
    return


# transformer

def bbox_rel_to_abs(bbox, pic_size):
    """
    :function: Transform bbox coordinates from relative to absolute.
    :param bbox: numpy.ndarray, (N, 4)
    :param pic_size: numpy.ndarray or tuple, (height, width)
    :return: absolute coordinates of bbox
    """
    height, width = pic_size
    abs_bbox = bbox * np.array([width, height] * 2)
    return abs_bbox


def bbox_abs_to_rel(bbox, pic_size):
    """
    :function: Transform absolute bbox coordinates to relative coordinates.
    :param bbox: numpy.adarray, uint8, (N, 4)
    :param pic_size: numpy.ndarray, (height, width)
    :return: relative coordiantes of bbox
    """
    height, width = pic_size
    rel_bbox = bbox.astype('float32') / np.array([width, height] * 2)
    return rel_bbox


def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), to_int=False):
    """
    :function: denormalize the image data.
    :param img: numpy.ndarray, (h, w, c)
    :param mean:
    :param std:
    :return: denormalized image
    """
    denorm_img = img * np.array(std) + np.array(mean)
    if to_int == True:
        denorm_img = denorm_img * 255
        denorm_img = denorm_img.astype('int8')
    return denorm_img


transformer = mx.gluon.data.vision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# transformer = mx.gluon.data.vision.transforms.Compose([
#     mx.gluon.data.vision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])


def transform_fn(*data):
    """
    almost the same with prepare_datum_for_net, and this function is for batchify.

    :param data: (img, label)
    :return: (img, label)
    """
    import copy
    img, label = copy.deepcopy(data)
    img_size = img.shape[:2]
    img = img.astype('float32')  # deepcopy
    label = label.astype('float32')

    img, _ = resize_img_and_label(img, label, size=300)
    img = mx.img.color_normalize(mx.nd.array(img), mean=mx.nd.array(mean), std=mx.nd.array(std))
    img = to_tensor(img)

    label[:, 1:] = bbox_abs_to_rel(label[:, 1:], img_size)
    label = mx.nd.array(label)

    return img, label


# validation

def validate(img, label, net, the_first_n_bboxes=3, std=(0.229, 0.224, 0.225),
             mean=(0.485, 0.456, 0.406)):
    """
    Inputs:

    img: numpy.array, shape (h, w, c)  without normalization
    label: np.array, shape (N, 5)
        n represents thd index of label boxes in one image.
    cls_preds: NDArray, shape (1, N, 2)
        N represents number of anchors. 2 represents 1 type of object
    anchors: NDArray, shape (1, N, 4)
        4 represents [rx1, ry1, rx2, ry2]. These are relative coordinates.
    bbox_preds: NDArray, shape (1, N*4)
        reshape(N, 4), equivalent to the shape of anchors

    Outputs:
        image with predicted bounding boxes.
    """
    height, width = img.shape[:2]
    import matplotlib.pyplot as plt
    fig = plt.imshow(img)
    axes = fig.axes

    mx_img, _ = resize_img_and_label(img, label)
    mx_img = mx.img.color_normalize(mx_img, mean=mean, std=std)
    mx_img = to_tensor(mx_img)
    mx_img = mx_img.expand_dims(axis=0)
    mx_img = mx_img.as_in_context(net.ctx)

    anchors, cls_preds, bbox_preds = net(mx_img)

    cls_probs = cls_preds.softmax().transpose(axes=(0, 2, 1))
    detect = mx.contrib.nd.MultiBoxDetection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(detect[0]) if row[0].asscalar() != -1]
    detection_output = detect[0, idx]
    print(detection_output)

    for relative_bbox in detection_output[0:the_first_n_bboxes, 2:6]:
        _add_rectangle(axes=axes, relative_bbox=relative_bbox.asnumpy(), color='blue')

    relative_label = bbox_abs_to_rel(label[0, 1:], np.array([height, width]))
    _add_rectangle(axes=axes, relative_bbox=relative_label, color='red')
    plt.show()


def validate_data_n (n, dataset, net, the_first_n_bboxes=3):
    """
    reset data_iter, and validate the n-th data. The index of data starts from 1.
    :param n:
    :param the_first_N_bboxes:
    :return:
    """

    img, label = dataset[n]
    validate(img, label, net, the_first_n_bboxes)