import cv2
import matplotlib.pyplot as plt

farina = cv2.imread("farina.png", 0) # 传入图像

hist_full = cv2.calcHist([farina], [0], None, [256], [0, 256])

plt.plot(hist_full)  # 查看直方图
plt.show()

"""
可以看到所有像素的灰度值大部分集中在20-50之间，这使得整个图像很暗淡。
也就是说对比度不高。如果我们通过灰度变换，将灰度值拉伸到整个0-255的区间，那么其对比度显然是大幅增强的。
可以用如下的公式来将某个像素的灰度值映射到更大的灰度空间：
"""
Imax = np.max(farina)
Imin = np.min(farina)
MAX = 255
MIN = 0
farina_cs = (farina - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
cv2.imshow("farina_cs", farina_cs.astype("uint8"))
cv2.waitKey()

