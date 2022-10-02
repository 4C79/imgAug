import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET


def rotate(imgs_path, xmls_path, img_save_path, xml_save_path):
    for images in os.listdir(imgs_path):
        # rotate_img
        oriname = images.rstrip('.jpg')
        img_path = os.path.join(imgs_path, oriname + '.jpg')
        img = cv2.imread(img_path)
        number = oriname.rsplit('_', 1)
        number[1] = 300 + int(number[1])
        H, W, C = img.shape
        # 旋转中心，逆时针旋转90度，最后一个是缩放因子
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 30, 1)
        dst = cv2.warpAffine(img, M, (W, H))
        cv2.imwrite(img_save_path + number[0] + '_' + str(number[1]) + '.jpg', dst)

        # rotate_xml
        xml_path = os.path.join(xmls_path, oriname + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find('filename').text
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for object in root.findall('object'):
            object_name = object.find('name').text
            Xmin = int(object.find('bndbox').find('xmin').text)
            Ymin = int(object.find('bndbox').find('ymin').text)
            Xmax = int(object.find('bndbox').find('xmax').text)
            Ymax = int(object.find('bndbox').find('ymax').text)
            # 修改属性
            w = Xmax - Xmin
            h = Ymax - Ymin
            object.find('bndbox').find('xmin').text = str(Ymin)
            object.find('bndbox').find('ymin').text = str(W - Xmax)
            object.find('bndbox').find('xmax').text = str(Ymax)
            object.find('bndbox').find('ymax').text = str(W - Xmin)
        size.find('width').text = str(H)
        size.find('height').text = str(W)
        root.find('filename').text = number[0] + '_' + str(number[1]) + '.jpg'

        tree.write(xml_save_path + number[0] + '_' + str(number[1]) + '.xml')
