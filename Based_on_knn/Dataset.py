import numpy
import os
from fnmatch import fnmatch
import cv2
class Dataset():
    # 初始化神经网络
    #filedir:文件路径
    #content: 'train'/'test'
    #num:数据数量
    #code_num:验证码数量
    #feature_set: 1 提取局部特征  0 不提取局部特征
    #feature_num: 特征数量
    #maxy:纵向上边界
    #miny:纵向下边界
    #minx:横向起始位置
    #maxx:横向结束位置
    #distance:每个字符相差距离
    def __init__(self,filedir,content,num,feature_set=1,feature_num=5,code_num=5,maxy=24,miny=5,maxx=17,minx=5,distance=24):
        self.filedir=filedir
        self.content=content
        self.num=num
        self.maxy=maxy
        self.miny=miny
        self.maxx=maxx
        self.minx =minx
        self.distance = distance
        self.feature_set=feature_set
        self.feature_num = feature_num
        self.code_num=code_num
    def feature(self, A):
        midx = int(A.shape[1] / 2) + 1
        midy = int(A.shape[0] / 2) + 1
        A1 = A[0:midy, 0:midx].mean()
        A2 = A[midy:A.shape[0], 0:midx].mean()
        A3 = A[0:midy, midx:A.shape[1]].mean()
        A4 = A[midy:A.shape[0], midx:A.shape[1]].mean()
        # A5=A.mean()
        A5 = A[midy - 1:midy + 2, midx - 1:midx + 2].mean()
        AF = [A1, A2, A3, A4, A5]
        return AF
    # 训练已知图片的特征
    def data(self):
        data_set = numpy.zeros(shape=(self.num * self.code_num, self.feature_num))
        k = 0
        label = []
        file_dir = self.filedir+self.content+'set'
        for file in os.listdir(file_dir):
            if fnmatch(file, '*.jpg'):
                img_name = file
                im = self._get_dynamic_binary_image(file_dir, img_name)
                im = self.interference_line(im)
                im = self.interference_point(im)
                for i in range(self.code_num):
                    data_set[k * self.code_num + i] = self.feature(im[self.miny:self.maxy, i * self.distance+ self.minx:i * self.distance+ self.maxx])
                    label.append(int(img_name.split('.')[0][i]))
                k = k + 1
        numpy.save(self.filedir+self.content+'_label.npy', label)
        print(self.content+'_label.npy'+'保存成功')
        numpy.save(self.filedir+self.content+'_set.npy', data_set)
        print(self.content + '_set.npy' + '保存成功')
    def interference_line(self,img):
        '''干扰线降噪'''
        h, w = img.shape[:2]
        for y in range(1, w - 1):
            for x in range(1, h - 1):
                count = 0
                if img[x, y - 1] > 245:
                    count = count + 1
                if img[x, y + 1] > 245:
                    count = count + 1
                if img[x - 1, y] > 245:
                    count = count + 1
                if img[x + 1, y] > 245:
                    count = count + 1
                if count > 2:
                    img[x, y] = 255
        return img
    def interference_point(self,img, x=0, y=0):
        """点降噪 9邻域框,以当前点为中心的田字框,黑点个数"""
        # todo 判断图片的长宽度下限
        cur_pixel = img[x, y]  # 当前像素点的值
        height, width = img.shape[:2]
        for y in range(0, width - 1):
            for x in range(0, height - 1):
                if y == 0:  # 第一行
                    if x == 0:  # 左上顶点,4邻域
                        # 中心点旁边3个点
                        sum = int(cur_pixel) \
                              + int(img[x, y + 1]) \
                              + int(img[x + 1, y]) \
                              + int(img[x + 1, y + 1])
                        if sum <= 2 * 245:
                            img[x, y] = 0
                    elif x == height - 1:  # 右上顶点
                        sum = int(cur_pixel) \
                              + int(img[x, y + 1]) \
                              + int(img[x - 1, y]) \
                              + int(img[x - 1, y + 1])
                        if sum <= 2 * 245:
                            img[x, y] = 0
                    else:  # 最上非顶点,6邻域
                        sum = int(img[x - 1, y]) \
                              + int(img[x - 1, y + 1]) \
                              + int(cur_pixel) \
                              + int(img[x, y + 1]) \
                              + int(img[x + 1, y]) \
                              + int(img[x + 1, y + 1])
                        if sum <= 3 * 245:
                            img[x, y] = 0
                elif y == width - 1:  # 最下面一行
                    if x == 0:  # 左下顶点
                        # 中心点旁边3个点
                        sum = int(cur_pixel) \
                              + int(img[x + 1, y]) \
                              + int(img[x + 1, y - 1]) \
                              + int(img[x, y - 1])
                        if sum <= 2 * 245:
                            img[x, y] = 0
                    elif x == height - 1:  # 右下顶点
                        sum = int(cur_pixel) \
                              + int(img[x, y - 1]) \
                              + int(img[x - 1, y]) \
                              + int(img[x - 1, y - 1])
                        if sum <= 2 * 245:
                            img[x, y] = 0
                    else:  # 最下非顶点,6邻域
                        sum = int(cur_pixel) \
                              + int(img[x - 1, y]) \
                              + int(img[x + 1, y]) \
                              + int(img[x, y - 1]) \
                              + int(img[x - 1, y - 1]) \
                              + int(img[x + 1, y - 1])
                        if sum <= 3 * 245:
                            img[x, y] = 0
                else:  # y不在边界
                    if x == 0:  # 左边非顶点
                        sum = int(img[x, y - 1]) \
                              + int(cur_pixel) \
                              + int(img[x, y + 1]) \
                              + int(img[x + 1, y - 1]) \
                              + int(img[x + 1, y]) \
                              + int(img[x + 1, y + 1])
                        if sum <= 3 * 245:
                            img[x, y] = 0
                    elif x == height - 1:  # 右边非顶点
                        sum = int(img[x, y - 1]) \
                              + int(cur_pixel) \
                              + int(img[x, y + 1]) \
                              + int(img[x - 1, y - 1]) \
                              + int(img[x - 1, y]) \
                              + int(img[x - 1, y + 1])
                        if sum <= 3 * 245:
                            img[x, y] = 0
                    else:  # 具备9领域条件的
                        sum = int(img[x - 1, y - 1]) \
                              + int(img[x - 1, y]) \
                              + int(img[x - 1, y + 1]) \
                              + int(img[x, y - 1]) \
                              + int(cur_pixel) \
                              + int(img[x, y + 1]) \
                              + int(img[x + 1, y - 1]) \
                              + int(img[x + 1, y]) \
                              + int(img[x + 1, y + 1])
                        if sum <= 4 * 245:
                            img[x, y] = 0
        return img

    def _get_dynamic_binary_image(self,file_dir, img_name):
        '''自适应阀值二值化'''
        img_name = file_dir + '/' + img_name
        im = cv2.imread(img_name)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        th1 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
        return th1
Test= Dataset('D:/ANN/DataSet/', 'test', 500)
Test.data()
