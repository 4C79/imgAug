# 深度学习数据集增强工具 

[TOC]



## 软件目的

- **通过软件的方式来快速增强深度学习中的数据集，从而减少人工成本，增强网络可靠性**
- **拟对图像数据增强的同时，相对应的改变其对应的annotation**
- **初步计划增强的数据格式为 coco，vot/c 数据集**

## 软件功能

**拟通过可选方式，可控数量来对特定文件夹下的图片进行增强**

**已实现的功能如下**

**增强中涉及到bbox变换的方式有：**

- **旋转，仿射变换**
- **翻转**
- **平移**
- **尺度变化**

**增强中不涉及到bbox变换的方式有：**

- **对比度、亮度变化**
- **噪声扰动**
- **颜色变化**
- **随机遮掩**

## 效果展示

### **原始图像**

![image-20220930102515274](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930102515274.png)

### **水平翻转**

```
将图像镜像翻转（水平）,参数表示翻转图片的概率
```

![image-20220930102801254](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930102801254.png)

### 垂直翻转

```
将图像镜像翻转（垂直）,参数表示翻转图片的概率
```

![image-20220930102935178](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930102935178.png)

### 反转颜色

```
将图像像素反转，即将像素变为 255-x ，参数表示图像转换的概率
```

![image-20220930103045701](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930103045701.png)

### 像素增减

```
变换图像中每个像素的像素值，参数表示增减多少，per_channel表示是否所有通道均变化
```

![image-20220930103313052](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930103313052.png)

### 图像压缩

```
压缩图像，值代表程度 0-100
```

![image-20220930103412141](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930103412141.png)

### 高斯模糊

```
对图像增加高斯模糊 , scale = 0.0 - 1
```

![image-20220930103548608](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930103548608.png)

### 像素丢失

```
随机丢失像素，第一个参数表示丢失的数量，第二个表示在分辨率为size_percent下进行丢失
```

![image-20220930103639041](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930103639041.png)

### 对比度增强

```
对比度增强，参数表示范围
```

![image-20220930103813302](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930103813302.png)

### 仿射变换

```
缩放，参数表示范围
```

<center class="half">
    <img src="C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930104438318.png" width="400"/><img src="C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930103948459.png" width="400"/>
</center>


### X轴平移

```
水平平移，参数表示程度
```



<center class="half">
    <img src="C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930104615287.png" width="400"/><img src="C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930104753233.png" width="400"/>
</center>


### Y轴平移

```
垂直方向移动，参数表示程度
```

<center class="half">
    <img src="C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930104857432.png" width="400"/><img src="C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930104907289.png" width="400"/>
</center>


### 变换方式可随意搭配

```
下图为选上除反色外所有变换
```

![image-20220930105145343](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930105145343.png)

```
每次可增强图片自定（这里为10张）
```

![image-20220930105226167](C:\Users\LRSJ\AppData\Roaming\Typora\typora-user-images\image-20220930105226167.png)

## 下一步计划

- **完善增强功能**
- **解决旋转的bbox变换**
- **优化代码结构**
- **写出合适的界面**
- **导出体量尽可能小的exe**