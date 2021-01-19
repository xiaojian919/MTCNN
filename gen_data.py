import os
from PIL import Image
import numpy as np
from tool import utils
import random
#输出详细异常信息
import traceback

anno_src = r"D:\程序代码\数据集\CelebA\CelebA\Anno\list_bbox_celeba.txt"
img_dir = r"D:\程序代码\数据集\CelebA\CelebA\Img\img_celeba.7z\img_celeba.7z\img_celeba"

save_path = r"E:\MTCNN\CelebA"


def gen_sample(face_size,stop_value):
    print("gen size:{} image" .format(face_size) )

    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")


    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0
    #这里readlines会有/n readline没有
    # test = open(positive_anno_filename, "r")
    # print(test.readline())
    # exit()

    try:
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue


            strs = line.split()
            #strs = ['000001.jpg', '95', '71', '226', '313']

            image_filename = strs[0].strip()
            print(image_filename)
            image_file = os.path.join(img_dir, image_filename)

            with Image.open(image_file) as img:
                img_w, img_h = img.size
                x1 = float(strs[1].strip())
                y1 = float(strs[2].strip())
                w = float(strs[3].strip())
                h = float(strs[4].strip())
                x2 = float(x1 + w)
                y2 = float(y1 + h)

                px1 = 0#float(strs[5].strip())
                py1 = 0#float(strs[6].strip())
                px2 = 0#float(strs[7].strip())
                py2 = 0#float(strs[8].strip())
                px3 = 0#float(strs[9].strip())
                py3 = 0#float(strs[10].strip())
                px4 = 0#float(strs[11].strip())
                py4 = 0#float(strs[12].strip())
                px5 = 0#float(strs[13].strip())
                py5 = 0#float(strs[14].strip())

                if x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue
                #原图的框框
                boxes = [[x1, y1, x2, y2]]

                cx = x1 + w / 2
                cy = y1 + h / 2
                side_len = max(w, h)



                count = 0
                for _ in range(5):
                    # 边长放缩比例
                    side_len_scale = np.random.uniform(0.8, 1.2)
                    _side_len = int(side_len * side_len_scale)
                    if _side_len == 0:
                        continue
                    # print(_side_len)

                    if np.random.randint(0, 10) <1:
                        #生成同一范围内两个随机数
                        #给xy不同比例的偏移量
                        positive_scale = np.random.uniform(0, 0.08,2)

                        # part_scale = np.random.uniform(0.15, 0.25)

                        part_scale = np.random.uniform(0.2, 0.6,2)
                        part_scale = part_scale if np.random.randint(0, 2) else -part_scale
                        offset_scale = [positive_scale, part_scale]
                        seed = offset_scale[np.random.randint(0, 2)]
                        _cx = cx + int(-_side_len * seed[0])
                        _cy = cy + int(-_side_len * seed[1])

                        _x1 = _cx - _side_len / 2
                        _y1 = _cy - _side_len / 2
                        _x2 = _x1 + _side_len
                        _y2 = _y1 + _side_len
                    else:
                        if img_w - _side_len <= 0 or img_h - _side_len <= 0:
                            continue
                        else:
                            _x1 = np.random.randint(0, img_w - _side_len)
                            _y1 = np.random.randint(0, img_h - _side_len)
                            _x2 = _x1 + _side_len
                            _y2 = _y1 + _side_len

                    if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
                        continue

                    offset_x1 = (x1 - _x1) / _side_len
                    offset_y1 = (y1 - _y1) / _side_len
                    offset_x2 = (x2 - _x2) / _side_len
                    offset_y2 = (y2 - _y2) / _side_len

                    offset_px1 = 0#(px1 - x1_) / side_len
                    offset_py1 = 0#(py1 - y1_) / side_len
                    offset_px2 = 0#(px2 - x1_) / side_len
                    offset_py2 = 0#(py2 - y1_) / side_len
                    offset_px3 = 0#(px3 - x1_) / side_len
                    offset_py3 = 0#(py3 - y1_) / side_len
                    offset_px4 = 0#(px4 - x1_) / side_len
                    offset_py4 = 0#(py4 - y1_) / side_len
                    offset_px5 = 0#(px5 - x1_) / side_len
                    offset_py5 = 0#(py5 - y1_) / side_len


                    crop_box = [_x1, _y1, _x2, _y2]
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size))

                    iou = utils.iou(crop_box, np.array(boxes))[0]
                    if iou > 0.65:
                        if positive_count >= stop_value*0.2:
                            continue
                        positive_anno_file.write(
                            "positive\{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                positive_count, 1, offset_x1, offset_y1,
                                offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        #flush 写入一次保存一次
                        positive_anno_file.flush()
                        face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                        positive_count += 1
                    elif 0.5>iou > 0.25:
                        if part_count >= stop_value*0.2:
                            continue
                        part_anno_file.write(
                            "part\{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                part_count, 2, offset_x1, offset_y1,offset_x2,
                                offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        part_anno_file.flush()
                        face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                        part_count += 1
                    elif iou < 0.1:
                        if negative_count >= stop_value*0.6:
                            continue
                        negative_anno_file.write(
                            "negative\{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(negative_count, 0))
                        negative_anno_file.flush()
                        face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                        negative_count += 1

                    count = positive_count+part_count+negative_count
                    print(positive_count,part_count,negative_count)
                if count>=stop_value:
                    break


    finally:
        positive_anno_file.close()
        negative_anno_file.close()
        part_anno_file.close()

if __name__ == '__main__':


    gen_sample(48, 100000)
    gen_sample(24, 100000)
    gen_sample(12, 100000)