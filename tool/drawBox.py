import cv2
import numpy as np
import xml.etree.ElementTree as ET

def dbox(img_path,xml_path):
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        class_name = object.find("name").text
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        # b_box 左上角坐标
        ptLeftTop = np.array([int(bndbox.find('xmin').text), int(bndbox.find('ymin').text)])
        # b_box 右下角坐标
        ptRightBottom = np.array([int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)])

    # 框的颜色
    point_color = (0, 255, 0)
    # 线的厚度
    thickness = 2
    # 线的类型
    lineType = 4

    # (500, 375, 3) -> h w c
    src = cv2.imread(img_path)
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    src = np.array(src)
    # 画 b_box
    cv2.rectangle(src, tuple(ptLeftTop), tuple(ptRightBottom), point_color, thickness, lineType)

    # 获取文字区域框大小
    t_size = cv2.getTextSize(class_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
    # 获取 文字区域右下角坐标
    textlbottom = ptLeftTop + np.array(list(t_size))
    # 绘制文字区域矩形框
    cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom), point_color, -1)
    # 计算文字起始位置偏移
    ptLeftTop[1] = ptLeftTop[1] + (t_size[1] / 2 + 4)
    # 绘字
    cv2.putText(src, class_name, tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    # 打印图片的shape
    print(src.shape)
    cv2.imwrite("test.jpg",src)

if __name__ == '__main__':
    dbox("apple_1.jpg","apple_1.xml")
