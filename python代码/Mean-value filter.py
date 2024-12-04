import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = r'D:\RJ\Pycharm\PyCharm Community Edition 2023.3.6\pycharmproject\pythonProject1\picture\p3.png'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# 判断图像是否为彩色图像（第三维大小为3表示彩色图像，有红、绿、蓝三个通道）
if len(image.shape) == 3 and image.shape[2] == 3:
    # 将彩色图像转换为灰度图像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
elif len(image.shape) == 3 and image.shape[2] == 4:  # 如果是RGBA图像
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

# 创建7x7的均值滤波核
h = np.ones((7, 7), dtype=np.float32) / 49

# 使用cv2.filter2D函数进行均值滤波
filtered_image = cv2.filter2D(image, -1, h)

# 显示原图像和滤波后的图像对比
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()