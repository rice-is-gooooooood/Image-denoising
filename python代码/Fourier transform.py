import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = r'D:\RJ\Pycharm\PyCharm Community Edition 2023.3.6\pycharmproject\pythonProject1\picture\p3.png'
img = cv2.imread(image_path)

# 判断图像是否为彩色图像（第三维大小为3表示彩色图像，有红、绿、蓝三个通道）
if img is not None and len(img.shape) == 3:
    # 将彩色图像转换为灰度图像
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    grayImg = img

# 执行傅里叶变换
F = np.fft.fft2(grayImg.astype(float))  # 对灰度图像进行二维傅里叶变换
F_shifted = np.fft.fftshift(F)           # 移动零频率分量到中心

# 计算幅度谱
magnitude = np.abs(F_shifted)            # 幅度谱
logMagnitude = np.log(1 + magnitude)     # 对数尺度

# 显示原图和傅里叶变换结果
plt.figure(figsize=(10, 5))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(grayImg, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示傅里叶变换的幅度谱
plt.subplot(1, 2, 2)
plt.imshow(logMagnitude, cmap='gray')
plt.title('Magnitude Spectrum')
plt.axis('off')

plt.show()