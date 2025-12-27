import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示警告
import matplotlib.pyplot as plt
import warnings
import os

# 忽略matplotlib的警告
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def cluster_and_merge_lanes(lines, image_shape, yellow_mask, white_mask):
    """
    将检测到的线段聚类并合并为主要车道线
    基于位置和颜色将线段分为左车道线、中间线和右车道线
    """
    if lines is None or len(lines) == 0:
        return None
    
    height, width = image_shape[:2]
    
    # 将线段按x坐标位置和颜色分类
    left_lines = []    # 左侧区域（白色）
    right_lines = []   # 右侧区域（白色）
    center_yellow = [] # 中间区域（黄色）
    center_white = []  # 中间区域（白色）
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算线段长度和斜率
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if x2 - x1 == 0:
            slope = 999  # 垂直线
        else:
            slope = (y2 - y1) / (x2 - x1)
        
        # 计算线段中点坐标
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        
        # 检查线段沿线的多个点的颜色（更可靠的判断）
        num_points = 5
        yellow_count = 0
        white_count = 0
        for i in range(num_points):
            t = i / (num_points - 1)
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            if 0 <= py < height and 0 <= px < width:
                if yellow_mask[py, px] > 128:
                    yellow_count += 1
                if white_mask[py, px] > 128:
                    white_count += 1
        
        # 如果超过一半的点在黄色区域，则认为是黄色线
        is_yellow = yellow_count >= num_points * 0.4
        is_white = white_count >= num_points * 0.4
        
        # 根据x坐标位置、颜色和斜率分类
        if mid_x < width * 0.35:  # 左侧区域
            # 左侧车道线应该有负斜率（从左上到右下）
            if is_white and slope < -0.3:
                left_lines.append((line, length, slope, mid_x))
        elif mid_x > width * 0.65:  # 右侧区域
            # 右侧车道线应该有正斜率（从左下到右上）
            if is_white and slope > 0.3:
                right_lines.append((line, length, slope, mid_x))
        else:  # 中间区域（0.35-0.65）
            # 中间线可能是接近垂直的或者有轻微斜率
            if is_yellow:
                center_yellow.append((line, length, slope, mid_x))
            elif is_white:
                center_white.append((line, length, slope, mid_x))
    
    # 对每组线段进行斜率聚类，选择主要方向的线段
    def select_best_by_slope(line_group, max_lines=1):
        if len(line_group) == 0:
            return []
        
        # 按斜率聚类（使用中位数斜率作为参考）
        slopes = [l[2] for l in line_group]
        median_slope = np.median(slopes)
        
        # 选择斜率接近中位数的线段
        filtered = []
        for line_data in line_group:
            line, length, slope, mid_x = line_data
            # 斜率差异在30%以内
            if abs(slope - median_slope) / (abs(median_slope) + 0.01) < 0.3:
                filtered.append(line_data)
        
        if len(filtered) == 0:
            filtered = line_group
        
        # 从过滤后的线段中选择最长的
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [l[0] for l in filtered[:max_lines]]
    
    # 选择每组中最好的线段
    best_left = select_best_by_slope(left_lines, max_lines=1)
    best_right = select_best_by_slope(right_lines, max_lines=1)
    # 中间区域：优先黄色，否则选白色
    best_center = select_best_by_slope(center_yellow, max_lines=1)
    if len(best_center) == 0:
        best_center = select_best_by_slope(center_white, max_lines=1)
    
    print(f"  左侧白线(负斜率): {len(left_lines)}条 → 选择{len(best_left)}条")
    print(f"  中间黄线: {len(center_yellow)}条")
    print(f"  中间白线: {len(center_white)}条")
    print(f"  中间总选择: {len(best_center)}条")
    print(f"  右侧白线(正斜率): {len(right_lines)}条 → 选择{len(best_right)}条")
    
    # 合并所有选中的线段
    final_lines = best_left + best_center + best_right
    
    return final_lines if len(final_lines) > 0 else None

def filter_lane_lines(lines, image_shape):
    """
    根据车道线的几何特征过滤直线
    车道线特点：
    1. 角度在一定范围内（通常30°-150°，避免水平线）
    2. 长度要足够长
    3. 位置主要在图像下半部分
    """
    if lines is None:
        return None
    
    height, width = image_shape[:2]
    filtered_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 计算线段长度
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 计算角度（相对于水平线）
        if x2 - x1 != 0:
            angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
        else:
            angle = 90
        
        # 过滤条件：
        # 1. 长度大于40像素（降低要求）
        # 2. 角度在20-88度之间（扩大角度范围）
        # 3. 线段主要在图像下半部分（y坐标较大）
        if length > 40 and 20 < angle < 88:
            # 至少有一个端点在图像下半部分
            if y1 > height * 0.25 or y2 > height * 0.25:  # 从0.3降至0.25
                filtered_lines.append(line)
    
    return filtered_lines if len(filtered_lines) > 0 else None

def process_lane_keep_largest(image_path):
    """
    车道线检测函数
    核心改进：
    1. 使用LAB和HLS双色彩空间提取白色和黄色
    2. 自适应阈值处理
    3. 合理的ROI区域
    4. 基于角度和位置的线段过滤
    """
    # 第一步：图像读取与预处理
    image = cv2.imread(image_path)
    if image is None: 
        print(f"无法读取图像: {image_path}")
        return
    height, width = image.shape[:2]
    
    # 第二步：多通道车道线提取（白色+黄色）
    # 转换色彩空间
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 提取白色车道线（使用L通道和灰度图）- 降低阈值以检测更多白色区域
    l_channel = lab[:,:,0]
    _, white_mask1 = cv2.threshold(l_channel, 180, 255, cv2.THRESH_BINARY)  # 从200降至180
    _, white_mask2 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)       # 从160降至140
    white_mask = cv2.bitwise_or(white_mask1, white_mask2)
    
    # 额外的白色增强：使用Sobel算子检测边缘
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    _, sobel_mask = cv2.threshold(scaled_sobel, 30, 255, cv2.THRESH_BINARY)
    white_mask = cv2.bitwise_or(white_mask, sobel_mask)
    
    # 提取黄色车道线（使用HLS空间 - 放宽阈值）
    lower_yellow = np.array([15, 30, 80], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    # 对黄色掩膜进行膨胀，扩大黄色区域以便更好地匹配线段
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.dilate(yellow_mask, kernel_dilate, iterations=2)
    
    # 合并白色和黄色掩膜
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    
    # 第三步：CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(combined_mask)
    
    # 第四步：ROI区域（梯形）
    mask = np.zeros_like(enhanced)
    roi_vertices = np.array([
        [
            (0, height),                              # 左下角
            (width, height),                          # 右下角
            (int(width * 0.70), int(height * 0.35)), # 右上角
            (int(width * 0.30), int(height * 0.35))  # 左上角 
        ]
    ], dtype=np.int32)
    
    cv2.fillPoly(mask, roi_vertices, 255)
    roi_masked = cv2.bitwise_and(enhanced, mask)
    
    # 第五步：形态学处理
    # 轻度开运算去除小噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(roi_masked, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 垂直方向闭运算，连接断裂的车道线
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # 第六步：Canny边缘检测（直接在处理后的图像上进行）
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)
    
    # 第七步：霍夫直线检测（调整参数以适应不同场景）
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=10,        # 进一步降低阈值，检测更多候选线段
        minLineLength=25,    # 降低最小长度
        maxLineGap=250       # 允许更大间隙，连接虚线
    )
    
    # 第八步：根据车道线几何特征过滤
    filtered_lines = filter_lane_lines(lines, image.shape)
    
    # 第九步：聚类并选择主要车道线（结合颜色信息）
    final_lines = cluster_and_merge_lanes(filtered_lines, image.shape, yellow_mask, white_mask)
    
    # 第十步：绘制结果
    result_img = image.copy()
    line_image = np.zeros_like(image)
    
    if final_lines is not None:
        print(f"检测到 {len(lines)} 条原始线段，过滤后保留 {len(filtered_lines) if filtered_lines else 0} 条，聚类后保留 {len(final_lines)} 条主要车道线")
        for line in final_lines:
            x1, y1, x2, y2 = line[0]
            # 在单独的图层上绘制，使用更粗的线条
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # 叠加到原图
        result_img = cv2.addWeighted(result_img, 0.8, line_image, 1.0, 0)
    else:
        print("未检测到符合条件的车道线")
    
    # 可视化展示关键步骤
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 4, 1)
    plt.title("Original Image", fontsize=10)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.title("White Mask", fontsize=10)
    plt.imshow(white_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.title("Yellow Mask", fontsize=10)
    plt.imshow(yellow_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.title("Combined Mask", fontsize=10)
    plt.imshow(combined_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.title("ROI Region", fontsize=10)
    plt.imshow(roi_masked, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.title("Morphology", fontsize=10)
    plt.imshow(closed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.title("Edges", fontsize=10)
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.title("Final Result", fontsize=10)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像而不是显示，避免警告
    output_dir = os.path.join(os.path.dirname(image_path), '..', 'output', 'exp2')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存可视化大图
    output_path = os.path.join(output_dir, 'lane_detection_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")
    plt.close()
    
    # 单独保存最终的车道检测结果图
    result_output_path = os.path.join(output_dir, 'lane_detection_result.png')
    cv2.imwrite(result_output_path, result_img)
    print(f"车道检测结果图已保存到: {result_output_path}")

# 测试车道线检测
if __name__ == "__main__":
    image_path = r'C:\Users\zxnh0\Desktop\CV_Exp\exp2_data\1.jpg'
    print(f"处理图像: {image_path}")
    process_lane_keep_largest(image_path)