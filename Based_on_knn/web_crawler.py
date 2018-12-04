# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:04:54 2017
@author: sl
"""
import requests
import time
def downloads_pic(pic_name):
         url='https://cas.ncu.edu.cn:8443/cas/codeimage;jsessionid=6C4FF6BDF2220560E083960675AE561F'
         res=requests.get(url,stream=True)  ####在罕见的情况下你可能想获取来自服务器的原始套接字响应，那么你可以访问 r.raw如果你确实想这么干，那请你确保在初始请求中设置了stream=True
         print(res)
         with open(r'D:\DATA\%s.jpg'%(str(pic_name)),'wb') as f:
                   print(res.iter_content(chunk_size=1024))
                   for chunk in res.iter_content(chunk_size=1024):  ####使用Response.iter_content将会处理大量你直接使用Response.raw不得不处理的.当流下载时，上面是优先推荐的获取内容方式
                            print(chunk)
                            if chunk: ###过滤下保持活跃的新块                      
                                     f.write(chunk)
                                     f.flush() #方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入
                   f.close()
if __name__=='__main__':
         for i in range(5):
                   pic_name=int(time.time()*1000000) #返回当前时间的时间戳（1970纪元后经过的浮点秒数）
                   downloads_pic(pic_name)
