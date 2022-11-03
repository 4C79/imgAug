import imgaug as ia
import numpy as np
import os
import shutil
import time
import xml.etree.ElementTree as ET
from PIL import Image
from imgaug import augmenters as iaa
from tqdm import tqdm

ia.seed(1)


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    # 修改增强后的xml文件中的filename
    elem = tree.find('filename')
    elem.text = (str(id) + '.jpg')
    xmlroot = tree.getroot()
    # 修改增强后的xml文件中的path
    elem = tree.find('path')
    if elem != None:
        elem.text = (saveroot + str(id) + '.jpg')

    index = 0
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # xmin = int(bndbox.find('xmin').text)
        # xmax = int(bndbox.find('xmax').text)
        # ymin = int(bndbox.find('ymin').text)
        # ymax = int(bndbox.find('ymax').text)

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(id + '.xml')))


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


# 选择img，annotation的源地与目的地,增强方式，增强数量
def imgaug(origin_path, save_path, tfList, valueList, AUGLOOP):
    IMG_DIR = origin_path + "//Images"
    XML_DIR = origin_path + "//Annotations"
    AUG_IMG_DIR = save_path + "//Images"
    AUG_XML_DIR = save_path + "//Annotations"
    boxes_img_aug_list = []
    new_bndbox_list = []
    tmp_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    msg_list = str(tmp_time) + " 此次图像增强选择的方式有: "

    # 翻转控制
    c_contrast_horizon = tfList[6]
    c_contrast_vertical = tfList[1]
    # 反色变换
    c_invert = tfList[4]
    # 像素增减
    c_increase = tfList[8]
    # 图像压缩
    c_jpegCompression = tfList[2]
    # 高斯模糊
    c_GaussianNoise = tfList[0]
    # 丢弃像素
    c_CoarseDropout = tfList[5]
    # 对比度增强
    c_ContrastNormalization = tfList[7]
    # 仿射变换
    c_Affine = tfList[10]
    # X轴平移
    c_translate_x = tfList[3]
    # Y轴平移
    c_translate_y = tfList[9]
    # 图像旋转
    c_rotation = tfList[11]

    # 解析源文件
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1

    mkdir(AUG_IMG_DIR)

    # 队列初始化
    seq = iaa.Sequential()

    # 增强控制

    # 将图像镜像翻转（水平）,参数表示翻转图片的概率 0-1
    if c_contrast_horizon == True:
        seq.append(iaa.Sequential([iaa.Fliplr(float(valueList[6]))]))
        msg_list += str(float(valueList[6])) + "的概率进行图像镜像（水平）翻转,"
    # 将图像镜像翻转（垂直）,参数表示翻转图片的概率 0-1
    if c_contrast_vertical == True:
        seq.append(iaa.Sequential([iaa.Flipud(float(valueList[1]))]))
        msg_list += str(float(valueList[1])) + "的概率进行图像镜像（垂直）翻转,"
    # 将图像像素反转，即将像素变为 255-x ，参数表示图像转换的概率 0-1
    if c_invert == True:
        seq.append(iaa.Sequential([iaa.Invert(float(valueList[4]))]))
        msg_list += str(float(valueList[4])) + "的概率进行像素反转,"
    # 变换图像中每个像素的像素值，参数表示增减多少（-20，20），per_channel表示是否所有通道均变化
    if c_increase == True:
        seq.append(iaa.Sequential([iaa.Add(value=(valueList[8], -valueList[8]), per_channel=True)]))
        msg_list += "像素在（-" + str(abs(valueList[8])) + "," + str(abs(valueList[8])) + ")的范围内随机增减,"
    # 压缩图像，值代表程度 0-100
    if c_jpegCompression == True:
        seq.append(iaa.Sequential([iaa.JpegCompression(compression=(0, valueList[2]))]))
        msg_list += "在（0" + "," + str(abs(valueList[2])) + "%)的范围内随机进行压缩,"
    # 对图像增加高斯模糊 , scale = 0.0 - 1
    if c_GaussianNoise == True:
        seq.append(iaa.Sequential([iaa.AdditiveGaussianNoise(scale=float(valueList[0]) * 255)]))
        msg_list += "以" + str(float(valueList[0])) + "的程度进行高斯图像模糊,"
    # 随机丢失像素，第一个参数表示丢失的数量，第二个表示在分辨率为size_percent下进行丢失
    if c_CoarseDropout == True:
        seq.append(iaa.Sequential([iaa.CoarseDropout((0, float(valueList[5])), size_percent=0.5)]))
        msg_list += "在（0" + "," + str(float(valueList[5])) + ")的范围内随机丢失图像,"
    # 对比度增强，范围在 0.5 - 1.5 之间
    if c_ContrastNormalization == True:
        print(valueList[7])
        seq.append(iaa.Sequential([iaa.ContrastNormalization((-float(valueList[7]), float(valueList[7])))]))
        msg_list += "像素在（-" + str(abs(valueList[7])) + "," + str(abs(valueList[7])) + ")的范围内随机增强对比度,"
    # 缩放，范围在 0.5 - 1.5 之间
    if c_Affine == True:
        seq.append(iaa.Sequential([iaa.Affine(scale=(min(1, float(valueList[10])), max(1, float(valueList[10]))))]))
        if float(valueList[10]) < 1:
            msg_list += "在" + str(min(1, float(valueList[10]))) + "," + str(
                max(1, float(valueList[10]))) + ")的范围内随机缩小图像,"
        else:
            msg_list += "在" + str(min(1, float(valueList[10]))) + "," + str(
                max(1, float(valueList[10]))) + ")的范围内随机放大图像,"
    # 平移 X 轴
    if c_translate_x == True:
        seq.append(iaa.Sequential([iaa.Affine(translate_percent={"x": (float(valueList[3]))})]))
        if float(valueList[3]) > 0:
            msg_list += "向右移动图像" + str(abs(valueList[3]) * 100) + "%,"
        else:
            msg_list += "向左移动图像" + str(abs(valueList[3]) * 100) + "%,"
    # 平移 Y 轴
    if c_translate_y == True:
        seq.append(iaa.Sequential([iaa.Affine(translate_percent={"y": (float(valueList[9]))})]))
        if float(valueList[9]) > 0:
            msg_list += "向下移动图像" + str(abs(valueList[9]) * 100) + "%,"
        else:
            msg_list += "向上移动图像" + str(abs(valueList[9]) * 100) + "%,"
    # 旋转图像，范围在 -180 - 180 之间
    if c_rotation == True:
        seq.append(iaa.Sequential([iaa.Affine(rotate=float(valueList[11]))]))
        msg_list += "顺时针旋转图像" + str(float(valueList[11])) + "°,"

    for name in tqdm(os.listdir(XML_DIR), desc='Processing'):

        bndbox = read_xml_annotation(XML_DIR, name)

        # 保存原xml文件
        shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
        # 保存原图
        og_img = Image.open(IMG_DIR + '/' + name[:-4] + '.jpg')
        og_img.convert('RGB').save(AUG_IMG_DIR + name[:-4] + '.jpg', 'JPEG')
        og_xml = open(os.path.join(XML_DIR, name))
        tree = ET.parse(og_xml)
        # 修改增强后的xml文件中的filename
        elem = tree.find('filename')
        elem.text = (name[:-4] + '.jpg')
        tree.write(os.path.join(AUG_XML_DIR, name))

        for epoch in range(AUGLOOP):
            seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
            # 读取图片
            img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
            # sp = img.size
            img = np.asarray(img)
            # bndbox 坐标增强
            for i in range(len(bndbox)):
                bbs = ia.BoundingBoxesOnImage([
                    ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                ], shape=img.shape)

            bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
            boxes_img_aug_list.append(bbs_aug)

            # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
            n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
            n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
            n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
            n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
            if n_x1 == 1 and n_x1 == n_x2:
                n_x2 += 1
            if n_y1 == 1 and n_y2 == n_y1:
                n_y2 += 1
            if n_x1 >= n_x2 or n_y1 >= n_y2:
                print('error', name)
            new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
            # 存储变化后的图片
            image_aug = seq_det.augment_images([img])[0]
            path = os.path.join(AUG_IMG_DIR,
                                str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
            image_auged = bbs.draw_on_image(image_aug, size=0)
            Image.fromarray(image_auged).convert('RGB').save(path)

            # 存储变化后的XML
            change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                       str(name[:-4]) + '_' + str(epoch))
            # print(str(str(name[:-4]) + '_' + str(epoch)) + '.jpg')
            new_bndbox_list = []

    msg_list += "\n\t此次运行共增强" + str(AUGLOOP) + "张图像."
    return msg_list


if __name__ == "__main__":
    IMG_DIR = "../data/Images/"
    XML_DIR = "../data/Annotations/"
    AUG_XML_DIR = "../AUG/Annotations/"  # 存储增强后的XML文件夹路径
    AUG_IMG_DIR = "../AUG/Images/"  # 存储增强后的影像文件夹路径
    angle = 360
    imgaug(IMG_DIR, XML_DIR, AUG_IMG_DIR, AUG_XML_DIR)

    # rotation2.rotate(IMG_DIR, XML_DIR, AUG_IMG_DIR, AUG_XML_DIR)
    print('Finish!')
