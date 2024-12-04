import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# 读取图像
image_path = r'D:\RJ\Pycharm\PyCharm Community Edition 2023.3.6\pycharmproject\pythonProject1\picture\p1.png'
img = cv2.imread(image_path)

# 如果是彩色图像，则转换颜色空间从 BGR 到 RGB（因为 OpenCV 默认读取为 BGR）
if img is not None and len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 显示原始图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

# 向图像添加高斯噪声（均值0，方差0.01）
noisy_img = random_noise(img, mode='gaussian', mean=0, var=0.01)

# 将 noisy_img 转换回 uint8 类型（random_noise 返回的是 float64 类型）
noisy_img = (255 * noisy_img).astype(np.uint8)

# 显示添加噪声后的图像
plt.subplot(1, 2, 2)
plt.imshow(noisy_img)
plt.title('Image with Gaussian Noise')
plt.axis('off')

plt.show()