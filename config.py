import os
import numpy as np

image_dir = '/Users/wuyingtianhua/Desktop/电信项目/3维重建/images/'
MRT = 0.7
#相机内参矩阵,其中，K[0][0]和K[1][1]代表相机焦距，而K[0][2]和K[1][2]
#代表图像的中心像素。
K = np.array([
        [2362.12, 0, 720],
        [0, 2362.12,  578],
        [0, 0, 1]])

#选择性删除所选点的范围。
x = 0.5
y = 1