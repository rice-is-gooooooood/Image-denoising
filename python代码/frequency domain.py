import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 读取图像
image_path = r'D:\RJ\Pycharm\PyCharm Community Edition 2023.3.6\pycharmproject\pythonProject1\picture\p3.png'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# 判断图像是否为彩色图像（第三维大小为3表示彩色图像，有红、绿、蓝三个通道）
if len(image.shape) == 3 and image.shape[2] == 3:
    # 将彩色图像转换为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
elif len(image.shape) == 3 and image.shape[2] == 4:  # 如果是RGBA图像
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

# 进行二维离散傅里叶变换
F = np.fft.fft2(image)
# 将零频率分量移到频谱中心
F_shifted = np.fft.fftshift(F)

# 获取图像的尺寸信息（行数和列数）
rows, cols = image.shape

# 计算频谱中心坐标
crow, ccol = rows // 2, cols // 2

# 定义滤波器半径（用于控制保留的低频范围，可根据实际需求调整）
radius = 40

# 创建与频谱大小相同的全零矩阵，用于构建滤波器
H = np.zeros((rows, cols))

# 循环构建理想低通滤波器（以频谱中心为圆心，radius为半径的圆形区域内为1，其余为0）
for r in range(rows):
    for c in range(cols):
        distance = np.sqrt((r - crow)**2 + (c - ccol)**2)
        if distance <= radius:
            H[r, c] = 1

# 在频率域对频谱进行滤波，即将频谱与滤波器相乘
F_filtered = F_shifted * H

# 将零频率分量移回原来的位置（恢复到和fft2变换前对应）
F_ishifted = np.fft.ifftshift(F_filtered)

# 进行二维离散傅里叶逆变换
image_processed = np.fft.ifft2(F_ishifted)

# 取实部（因为经过逆变换后结果可能存在微小的虚部，通常取实部作为最终图像数据）
image_processed = np.abs(image_processed)

# 显示原图像和处理后的图像对比
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_processed, cmap='gray')
plt.title('Processed Image')
plt.axis('off')

plt.show()