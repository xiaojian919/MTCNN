import numpy as np


def iou(box,boxes, isMin = False):
    """
    计算矩形框iou 需要切片，输入需要numpy格式
    :param box:初始矩形框
    :param boxes:与初始矩形框比较的矩形框组 可以输入一个或多个矩形框
    :param isMin: 是否包含 这里是在o网络中出现大框套小框 要把小框删除
    :return: boxes 对 box的iou
    """
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:
        ovr = np.true_divide(inter, np.minimum(box_area, area))
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr

def nms(boxes, thresh=0.3, isMin = False):#
    '''
    非极大值抑制
    :param boxes: 带置信度的一组框
    :param thresh: iou筛选阈值
    :param isMin: 同上 感觉没啥用
    :return: 删选过后的框
    '''
    # print(boxes.shape[0])
    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]
    # print(_boxes)
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]

        r_boxes.append(a_box)

        # print(iou(a_box, b_boxes))

        index = np.where(iou(a_box, b_boxes ,isMin)< thresh)#
        #这里返回的index是iou(a_box, b_boxes)这个数组的index
        #再将这个index用到b_boxes上 取出对应位置的框和置信度
        _boxes = b_boxes[index]

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    # return np.stack(r_boxes)
    return np.array(r_boxes)

def convert_to_square(bbox):
    """
    转正方形
    :param bbox: 矩形框组

    :return: 按最长边为边长的正方形框
    """
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)
    #这里如果超出范围裁剪会填充黑边
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox

if __name__ == '__main__':
    # a = np.array([1,1,11,11])
    # bs = np.array([[1,1,10,10],[14,15,20,20]])
    # print(iou(a,bs))

    bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    print((-bs[:,4]).argsort())
    print(nms(bs))
