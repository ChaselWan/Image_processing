# writer:wojianxinygcl@163.com

# date  : 2020.3.30 

# 原文地址 : https://www.cnblogs.com/wojianxin/p/12530172.html


import cv2
import numpy as np
from matplotlib import pyplot as plt


def fft_cv(img):
    """
    用cv进行傅里叶变换
    :param img:
    :return:
    """

    # 傅里叶变换

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    dftshift = np.fft.fftshift(dft)

    res1 = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))

    # 傅里叶逆变换

    ishift = np.fft.ifftshift(dftshift)

    iimg = cv2.idft(ishift)

    res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    # 显示图像

    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')

    plt.axis('off')

    plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('Fourier Image')

    plt.axis('off')

    plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('Inverse Fourier Image')

    plt.axis('off')

    plt.show()


def fft_numpy(img):
    """
    用numpy对图像进行傅里叶变换
    """
    # 傅里叶变换

    f = np.fft.fft2(img)

    fshift = np.fft.fftshift(f)

    res = np.log(np.abs(fshift))

    # 傅里叶逆变换

    ishift = np.fft.ifftshift(fshift)

    iimg = np.fft.ifft2(ishift)

    iimg = np.abs(iimg)

    # 展示结果

    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')

    plt.axis('off')

    plt.subplot(132), plt.imshow(res, 'gray'), plt.title('Fourier Image')

    plt.axis('off')

    plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')

    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    # 读取图像
    img = cv2.imread('LENA256.BMP', 0)  # 读取灰度图
    fft_numpy(img)  # 使用numpy进行变换
    # fft_cv(img)   # 使用cv进行变化

