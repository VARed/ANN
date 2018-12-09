import numpy
import os
from fnmatch import fnmatch
import cv2
class Dataset():
    # 采集数据
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
    def __init__(self,filedir,num,feature_set=1,feature_num=5):
        self.filedir=filedir
        self.num=num
        self.feature_set=feature_set
        file_dir = self.filedir + 'Data'
        for file in os.listdir(file_dir):
            if fnmatch(file, '*.jpg'):
                img_name = file
                im = self._get_dynamic_binary_image(file_dir, img_name)
                break
        h, w = im.shape[:2]
        x = numpy.zeros(shape=(w,))
        y = numpy.zeros(shape=(h,))
        xstart,xend,ystart,yend=[],[],[],[]
        k,flag= 0,0
        for file in os.listdir(file_dir):
            if fnmatch(file, '*.jpg'):
                img_name = file
                im = self._get_dynamic_binary_image(file_dir, img_name)
                im = self.clear_border(im)
                im = self.interference_line(im)
                im = 1-im/255
                k=k+1
                if k>30:
                    break
                for i in range(w):
                    x[i] = im[:, i].sum() + x[i]
                for i in range(h):
                    y[i] = im[i].sum() + y[i]
        for i in range(1,len((x> 80))-1):
            if (x > 80)[i] and flag==0:
                xstart.append(i)
                flag = 1
            if ~((x > 80)[i]):
                if flag:
                    xend.append(i)
                flag = 0
        flag=0
        for i in range(1,(len(( y> 100))-1)):
            if (y > 100)[i] and flag == 0:
                ystart.append(i)
                flag = 1
            if ~((y > 100)[i]):
                if flag:
                    yend.append(i)
                flag = 0
        self.maxy=int(yend[0])
        self.miny=int(ystart[0])
        self.maxx=int(xend[0])
        self.minx =int(xstart[0])
        self.distance = int(xstart[1]-xstart[0])
        self.code_num = len(xstart)
        if feature_set==0:
            self.feature_num=(self.maxy-self.miny)*(self.maxx-self.minx)
        else:
            self.feature_num = feature_num
    #特征提取
    def feature(self, A):
        midx = int(A.shape[1] / 2) + 1
        midy = int(A.shape[0] / 2) + 1
        A1 = A[0:midy, 0:midx].mean()
        A2 = A[midy:A.shape[0], 0:midx].mean()
        A3 = A[0:midy, midx:A.shape[1]].mean()
        A4 = A[midy:A.shape[0], midx:A.shape[1]].mean()
        A5 = A[midy - 1:midy + 2, midx - 1:midx + 2].mean()
        AF = [A1, A2, A3, A4, A5]
        return AF
    # 训练已知图片的特征
    def data(self):
        data_set = numpy.zeros(shape=(self.num * self.code_num, self.feature_num))
        k = 0
        label = []
        file_dir = self.filedir+'Data'
        for file in os.listdir(file_dir):
            if fnmatch(file, '*.jpg'):
                img_name = file
                im = self._get_dynamic_binary_image(file_dir, img_name)
                im = self.clear_border(im)
                im = self.interference_line(im)
                for i in range(self.code_num):
                    if self.feature_set:
                        data_set[k * self.code_num + i] = self.feature(im[self.miny:self.maxy, i * self.distance+ self.minx:i * self.distance+ self.maxx])
                    else:
                        data_set[k * self.code_num + i] = im[self.miny:self.maxy, i * self.distance + self.minx:i * self.distance + self.maxx].flatten()
                    label.append(int(img_name.split('.')[0][i]))
                k = k + 1
        numpy.save(self.filedir+'label.npy', label)
        print('label.npy'+'保存成功')
        self.normalize_dataset(data_set)
        numpy.save(self.filedir+'set.npy', data_set)
        print('set.npy' + '保存成功')
    # 归一化函数

    def normalize_dataset(self,data_set):
        for row in data_set:
            for i in range(len(row)):
                row[i] = row[i] / 255

    # 降噪

    def interference_line(self,img):
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

    def clear_border(self,img):
        '''去除边框'''
        h, w = img.shape[:2]
        for y in range(0, w):
            for x in range(0, h):
                if y < 4 or y > w - 4:
                    img[x, y] = 255
                if x < 4 or x > h - 4:
                    img[x, y] = 255
        return img
    def _get_dynamic_binary_image(self,file_dir, img_name):
        '''自适应阀值二值化'''
        img_name = file_dir + '/' + img_name
        im = cv2.imread(img_name)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        th1 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
        return th1
