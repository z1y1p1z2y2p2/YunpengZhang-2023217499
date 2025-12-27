# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_image(image_path):
    # 加载图像并转换为numpy数组
    img = Image.open(image_path)
    return np.array(img)

# 将RGB图像转换为灰度图
def rgb_to_gray(image):
    
    if len(image.shape) == 3:
        # 使用加权平均法：Gray = 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        return gray.astype(np.uint8)
    return image

# 实现2D卷积操作
def convolution_2d(image, kernel):  
    # 获取图像和卷积核的尺寸
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    # 计算padding大小
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # 对图像进行零填充
    padded_image = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w))
    padded_image[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = image
    
    # 初始化输出图像
    output = np.zeros((img_h, img_w))
    
    # 执行卷积操作
    for i in range(img_h):
        for j in range(img_w):
            # 提取当前窗口
            window = padded_image[i:i + k_h, j:j + k_w]
            # 计算卷积值
            output[i, j] = np.sum(window * kernel)
    
    return output

# 使用Sobel算子进行边缘检测
def sobel_filter(image):
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)
    
    # 转换为灰度图
    gray_image = rgb_to_gray(image).astype(np.float64)
    # 分别计算x和y方向的梯度
    gradient_x = convolution_2d(gray_image, sobel_x)
    gradient_y = convolution_2d(gray_image, sobel_y)
    # 计算边缘强度
    edge_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    # 归一化到0-255
    edge_magnitude = (edge_magnitude / edge_magnitude.max() * 255).astype(np.uint8)
    
    return edge_magnitude, gradient_x, gradient_y


# 使用给定卷积核进行滤波
def custom_kernel_filter(image, kernel):
    # 转换为灰度图
    gray_image = rgb_to_gray(image).astype(np.float64)
    # 执行卷积
    filtered = convolution_2d(gray_image, kernel)
    # 取绝对值并归一化到0-255
    filtered = np.abs(filtered)
    filtered = (filtered / filtered.max() * 255).astype(np.uint8)
    
    return filtered


def calculate_histogram(image):
    # 手动计算图像的颜色直方图
    if len(image.shape) == 3:
        # RGB图像
        histograms = {}
        channels = ['R', 'G', 'B']
        
        for i, channel in enumerate(channels):
            # 初始化直方图（256个bin）
            hist = np.zeros(256, dtype=np.int32)
            
            # 统计每个像素值的频次
            channel_data = image[:, :, i].flatten()
            for pixel_value in channel_data:
                hist[pixel_value] += 1
            
            histograms[channel] = hist
        return histograms
    
    else:
        # 灰度图像
        hist = np.zeros(256, dtype=np.int32)
        flat_image = image.flatten()
        for pixel_value in flat_image:
            hist[pixel_value] += 1
        return {'Gray': hist}


def visualize_histogram(histograms, save_path):
    # 可视化并保存颜色直方图
    if 'Gray' in histograms:
        # 灰度直方图
        plt.figure(figsize=(10, 6))
        plt.bar(range(256), histograms['Gray'], color='gray', width=1)
        plt.xlabel('像素值')
        plt.ylabel('频次')
        plt.title('灰度直方图')
        plt.xlim([0, 255])
    else:
        # RGB直方图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = ['red', 'green', 'blue']
        channels = ['R', 'G', 'B']
        titles = ['红色通道直方图', '绿色通道直方图', '蓝色通道直方图']
        
        for i, (channel, color, title) in enumerate(zip(channels, colors, titles)):
            axes[i].bar(range(256), histograms[channel], color=color, width=1, alpha=0.7)
            axes[i].set_xlabel('像素值')
            axes[i].set_ylabel('频次')
            axes[i].set_title(title)
            axes[i].set_xlim([0, 255])
        
        plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"直方图已保存至: {save_path}")

# 计算灰度共生矩阵 (Gray-Level Co-occurrence Matrix, GLCM)
def calculate_glcm(image, d=1, theta=0):
    # 确保是灰度图
    if len(image.shape) == 3:
        image = rgb_to_gray(image)
    
    # 量化灰度级到16级以减少计算量
    levels = 16
    image_quantized = (image / 256 * levels).astype(np.int32)
    image_quantized = np.clip(image_quantized, 0, levels - 1)
    
    # 初始化GLCM矩阵
    glcm = np.zeros((levels, levels), dtype=np.float64)
    
    # 根据角度确定偏移量
    if theta == 0:
        dx, dy = d, 0
    elif theta == 45:
        dx, dy = d, -d
    elif theta == 90:
        dx, dy = 0, -d
    elif theta == 135:
        dx, dy = -d, -d
    else:
        dx, dy = d, 0
    
    rows, cols = image_quantized.shape
    
    # 计算GLCM
    for i in range(rows):
        for j in range(cols):
            # 计算邻居位置
            ni, nj = i + dy, j + dx
            
            # 检查边界
            if 0 <= ni < rows and 0 <= nj < cols:
                # 获取当前像素和邻居像素的灰度值
                current_value = image_quantized[i, j]
                neighbor_value = image_quantized[ni, nj]
                
                # 更新GLCM
                glcm[current_value, neighbor_value] += 1
    
    # 使GLCM对称
    glcm = glcm + glcm.T
    
    # 归一化
    glcm_sum = glcm.sum()
    if glcm_sum > 0:
        glcm = glcm / glcm_sum
    
    return glcm


def extract_glcm_features(glcm):
    # 从GLCM中提取纹理特征
    features = {}
    levels = glcm.shape[0]
    
    # 创建坐标网格
    i_indices, j_indices = np.meshgrid(range(levels), range(levels), indexing='ij')
    
    # 计算均值和标准差
    mu_i = np.sum(i_indices * glcm)
    mu_j = np.sum(j_indices * glcm)
    
    sigma_i = np.sqrt(np.sum(((i_indices - mu_i) ** 2) * glcm))
    sigma_j = np.sqrt(np.sum(((j_indices - mu_j) ** 2) * glcm))
    
    # 1. 对比度 (Contrast): 衡量局部灰度变化
    features['contrast'] = np.sum(((i_indices - j_indices) ** 2) * glcm)
    
    # 2. 相关性 (Correlation): 衡量灰度线性依赖关系
    if sigma_i > 0 and sigma_j > 0:
        features['correlation'] = np.sum((i_indices - mu_i) * (j_indices - mu_j) * glcm) / (sigma_i * sigma_j)
    else:
        features['correlation'] = 0
    
    # 3. 能量 (Energy/ASM): 衡量灰度分布均匀程度
    features['energy'] = np.sum(glcm ** 2)
    
    # 4. 同质性 (Homogeneity): 衡量对角线附近元素的分布
    features['homogeneity'] = np.sum(glcm / (1 + np.abs(i_indices - j_indices)))
    
    # 5. 熵 (Entropy): 衡量图像的随机性
    # 避免log(0)
    glcm_nonzero = glcm.copy()
    glcm_nonzero[glcm_nonzero == 0] = 1e-10
    features['entropy'] = -np.sum(glcm * np.log2(glcm_nonzero))
    
    return features

 # 提取图像的纹理特征
def extract_texture_features(image):
    # 转换为灰度图
    gray_image = rgb_to_gray(image)
    
    # 在四个方向上计算GLCM
    angles = [0, 45, 90, 135]
    all_features = []
    
    for angle in angles:
        glcm = calculate_glcm(gray_image, d=1, theta=angle)
        features = extract_glcm_features(glcm)
        all_features.append(features)
    
    # 计算各方向特征的平均值
    avg_features = {}
    for key in all_features[0].keys():
        values = [f[key] for f in all_features]
        avg_features[key] = np.mean(values)
        avg_features[f'{key}_std'] = np.std(values)
    
    return avg_features

# 保存图像
def save_image(image, save_path):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        img = Image.fromarray(image)
        img.save(save_path)
        print(f"图像已保存至: {save_path}")


def main():
    # 设置路径
    input_path = 'exp1_data/1.jpg'
    output_dir = 'output/exp1'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载图像
    print("=" * 60)
    print("实验一：图像滤波")
    print("=" * 60)
    
    print("\n1. 加载输入图像...")
    image = load_image(input_path)
    print(f"   图像尺寸: {image.shape}")
    print(f"   图像数据类型: {image.dtype}")
    
    # 2. Sobel算子滤波
    print("\n2. 使用Sobel算子进行边缘检测...")
    sobel_result, gradient_x, gradient_y = sobel_filter(image)
    save_image(sobel_result, os.path.join(output_dir, 'sobel_filtered.jpg'))
    
    # 保存梯度图
    gx_normalized = ((gradient_x - gradient_x.min()) / (gradient_x.max() - gradient_x.min()) * 255).astype(np.uint8)
    gy_normalized = ((gradient_y - gradient_y.min()) / (gradient_y.max() - gradient_y.min()) * 255).astype(np.uint8)
    save_image(gx_normalized, os.path.join(output_dir, 'sobel_gradient_x.jpg'))
    save_image(gy_normalized, os.path.join(output_dir, 'sobel_gradient_y.jpg'))
    
    # 3. 给定卷积核滤波
    print("\n3. 使用给定卷积核进行滤波...")
    custom_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]], dtype=np.float64)
    print(f"   卷积核:\n{custom_kernel}")
    
    custom_result = custom_kernel_filter(image, custom_kernel)
    save_image(custom_result, os.path.join(output_dir, 'custom_kernel_filtered.jpg'))
    
    # 4. 计算并可视化颜色直方图
    print("\n4. 计算颜色直方图...")
    histograms = calculate_histogram(image)
    visualize_histogram(histograms, os.path.join(output_dir, 'color_histogram.png'))
    
    # 打印直方图统计信息
    for channel, hist in histograms.items():
        print(f"   {channel}通道 - 最大频次: {hist.max()}, 总像素数: {hist.sum()}")
    
    # 5. 提取纹理特征
    print("\n5. 提取纹理特征（GLCM）...")
    texture_features = extract_texture_features(image)
    
    print("   纹理特征：")
    for key, value in texture_features.items():
        print(f"   - {key}: {value:.6f}")
    
    # 保存纹理特征为npy格式
    texture_save_path = os.path.join(output_dir, 'texture_features.npy')
    np.save(texture_save_path, texture_features)
    print(f"\n   纹理特征已保存至: {texture_save_path}")
    
    # 6. 创建结果对比图
    print("\n6. 创建结果对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 灰度图
    gray_image = rgb_to_gray(image)
    axes[0, 1].imshow(gray_image, cmap='gray')
    axes[0, 1].set_title('灰度图像')
    axes[0, 1].axis('off')
    
    # Sobel滤波结果
    axes[0, 2].imshow(sobel_result, cmap='gray')
    axes[0, 2].set_title('Sobel边缘检测')
    axes[0, 2].axis('off')
    
    # 给定卷积核滤波结果
    axes[1, 0].imshow(custom_result, cmap='gray')
    axes[1, 0].set_title('给定卷积核滤波')
    axes[1, 0].axis('off')
    
    # X方向梯度
    axes[1, 1].imshow(gx_normalized, cmap='gray')
    axes[1, 1].set_title('Sobel X方向梯度')
    axes[1, 1].axis('off')
    
    # Y方向梯度
    axes[1, 2].imshow(gy_normalized, cmap='gray')
    axes[1, 2].set_title('Sobel Y方向梯度')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   对比图已保存至: {os.path.join(output_dir, 'comparison.png')}")
    
    print("\n" + "=" * 60)
    print("实验一完成！")
    print("=" * 60)
    
    return {
        'sobel_result': sobel_result,
        'custom_result': custom_result,
        'histograms': histograms,
        'texture_features': texture_features
    }

if __name__ == '__main__':
    main()
