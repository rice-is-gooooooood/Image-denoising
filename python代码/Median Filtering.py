import cv2
import matplotlib.pyplot as plt

# 读取图像
image_path = 'D:\RJ\Pycharm\PyCharm Community Edition 2023.3.6\pycharmproject\pythonProject1\picture\p3.png'
image = cv2.imread(image_path)

# 判断图像是否为彩色图像（第三维大小为3表示彩色图像，有红、绿、蓝三个通道）
if len(image.shape) == 3 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用中值滤波进行去噪，设置滤波核大小为3x3
filtered_image = cv2.medianBlur(image, 3)

# 显示原图像和滤波后的图像
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