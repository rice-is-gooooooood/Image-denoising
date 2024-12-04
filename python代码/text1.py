from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = 'D:\RJ\Pycharm\PyCharm Community Edition 2023.3.6\pycharmproject\pythonProject1\picture\p1.png'
image = Image.open(image_path).convert('L')  # 转换为灰度图像

# 将图像转换为NumPy数组
image_array = np.array(image)

# 获取图像尺寸
rows, cols = image_array.shape

# 设置椒盐噪声的密度（概率），这里设为0.05，表示有5%的像素点会变为噪声点
noise_density = 0.05

# 生成与图像大小相同的均匀分布随机数矩阵，用于确定哪些像素点变为噪声点
rand_matrix = np.random.rand(rows, cols)

# 遍历图像的每个像素点，添加椒盐噪声
for row in range(rows):
    for col in range(cols):
        # 根据噪声密度判断是否添加噪声
        if rand_matrix[row, col] < noise_density / 2:
            # 添加黑色噪声点（将像素值设为0，对于uint8类型图像）
            image_array[row, col] = 0
        elif rand_matrix[row, col] < noise_density:
            # 添加白色噪声点（将像素值设为255，对于uint8类型图像）
            image_array[row, col] = 255

# 将NumPy数组转换回PIL图像
noisy_image = Image.fromarray(image_array)

# 显示原图像和添加噪声后的图像对比
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Salt and Pepper Noise')
plt.axis('off')

plt.show()