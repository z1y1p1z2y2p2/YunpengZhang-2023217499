# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class CNN_Model(nn.Module):
    """
    改进的卷积神经网络模型
    使用BatchNorm和更深的结构提高鲁棒性
    """
    def __init__(self):
        super(CNN_Model, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 第一个卷积块
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.dropout1(self.pool1(x))
        
        # 第二个卷积块
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.dropout2(self.pool2(x))
        
        # 第三个卷积块
        x = self.relu(self.bn3(self.conv3(x)))
        
        # 全连接层
        x = self.flatten(x)
        x = self.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x


def load_mnist_data(batch_size=64):
    """加载MNIST数据集，使用强数据增强提高鲁棒性"""
    print("正在加载MNIST数据集...")
    
    # 强数据增强 - 模拟各种变形
    transform_train = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./mnist_data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root='./mnist_data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
    """训练模型"""
    print(f"\n开始训练模型（设备：{device}）...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        test_acc = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_acc)
        scheduler.step(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}] 完成 - 训练损失: {avg_loss:.4f}, 测试准确率: {test_acc:.2f}%')
    
    return train_losses, test_accuracies


def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


def preprocess_digit_image(image):
    """预处理单个数字图像为MNIST格式"""
    # 转灰度
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 对比度增强 - 对低对比度图像很有效
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(image)
    
    # 轻微模糊去噪
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 多种二值化方法组合
    _, binary1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    # 取交集获得更清晰的结果
    binary = cv2.bitwise_or(binary1, binary2)
    
    # 确保白字黑底（MNIST格式）
    if np.mean(binary) > 127:
        binary = 255 - binary
    
    # 轻微膨胀，填补细小断裂
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    # 找到数字边界并裁剪
    coords = cv2.findNonZero(binary)
    if coords is not None and len(coords) > 0:
        x, y, w, h = cv2.boundingRect(coords)
        # 稍微扩大边界
        pad = 2
        x = max(0, x - pad)
        y = max(0, y - pad)
        h_img, w_img = binary.shape
        w = min(w + 2*pad, w_img - x)
        h = min(h + 2*pad, h_img - y)
        binary = binary[y:y+h, x:x+w]
    
    h, w = binary.shape
    if h == 0 or w == 0:
        return torch.zeros(1, 1, 28, 28)
    
    # 保持长宽比缩放到20x20区域
    scale = min(20.0 / h, 20.0 / w)
    new_h = max(1, min(int(h * scale), 20))
    new_w = max(1, min(int(w * scale), 20))
    
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 居中放置在28x28画布
    final_img = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    final_img[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    # 归一化
    normalized = final_img.astype(np.float32) / 255.0
    normalized = (normalized - 0.1307) / 0.3081
    
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


def segment_digits_robust(image_path):
    """
    鲁棒的数字分割 - 处理有干扰和不规则排列
    使用多种策略尝试精确分割出10个数字
    """
    print(f"\n正在处理图像：{image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")
    
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    
    # 先进行对比度增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    best_candidates = []
    
    # 精简策略 - 3种最有效的分割策略
    strategies = [
        # 标准策略
        {'blur': 5, 'method': 'otsu', 'morph': True, 'min_w': 5, 'min_h': 12, 'min_area': 100, 'use_enhanced': True},
        # 适应性策略
        {'blur': 3, 'method': 'combined', 'morph': True, 'min_w': 3, 'min_h': 10, 'min_area': 60, 'use_enhanced': True},
        # 宽松策略（针对细笔画）
        {'blur': 3, 'method': 'adapt', 'morph': False, 'min_w': 2, 'min_h': 8, 'min_area': 40, 'use_enhanced': True},
    ]
    
    for idx, strategy in enumerate(strategies):
        input_img = enhanced if strategy.get('use_enhanced', False) else gray
        candidates = try_segmentation(input_img, img_h, img_w, strategy)
        
        # 如果找到正好10个，直接使用
        if len(candidates) == 10:
            best_candidates = candidates
            print(f"  策略{idx+1}成功：检测到10个数字")
            break
        
        # 记录最接近10的结果
        if abs(len(candidates) - 10) < abs(len(best_candidates) - 10) or len(best_candidates) == 0:
            best_candidates = candidates
    
    # 如果结果少于10个，尝试更宽松的检测
    if len(best_candidates) < 10:
        print(f"  当前检测到{len(best_candidates)}个，尝试补充检测...")
        # 尝试极宽松策略
        ultra_loose = {'blur': 3, 'method': 'adapt', 'morph': False, 
                      'min_w': 2, 'min_h': 8, 'min_area': 25, 'use_enhanced': True}
        candidates = try_segmentation(enhanced, img_h, img_w, ultra_loose)
        if len(candidates) >= len(best_candidates):
            best_candidates = candidates
    
    # 如果结果多于10个，智能筛选到10个
    if len(best_candidates) > 10:
        best_candidates = select_best_10(best_candidates, img_h)
    
    print(f"  最终检测到 {len(best_candidates)} 个数字区域")
    
    # 提取数字图像
    bboxes = [c['bbox'] for c in best_candidates]
    digit_images = []
    
    for x, y, w, h in bboxes:
        pad = max(3, int(min(w, h) * 0.15))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        digit_images.append(gray[y1:y2, x1:x2])
    
    return digit_images, bboxes, original


def try_segmentation(gray, img_h, img_w, strategy):
    """尝试一种分割策略"""
    blur_size = strategy['blur']
    if blur_size > 0:
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    else:
        blurred = gray.copy()
    
    # 二值化
    if strategy['method'] == 'otsu':
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif strategy['method'] == 'adapt':
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 5)
    else:  # combined
        _, b1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)
        binary = cv2.bitwise_or(b1, b2)  # 使用并集获得更完整的检测
    
    # 形态学清理
    if strategy['morph']:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 先闭运算连接断裂
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # 再开运算去噪
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 从策略获取阈值参数
    min_w = strategy.get('min_w', 8)
    min_h = strategy.get('min_h', 15)
    min_area = strategy.get('min_area', 150)
    
    # 收集候选区域
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        bbox_area = w * h
        
        if bbox_area == 0:
            continue
        
        fill_ratio = area / bbox_area
        aspect_ratio = h / float(w) if w > 0 else 0
        
        # 过滤条件 - 使用策略参数，特别宽松以检测数字1
        # 增加最大尺寸限制来排除整张图或大型干扰区域
        max_w = img_w * 0.20  # 单个数字宽度不应超过图像宽度的20%
        max_h = img_h * 0.60  # 单个数字高度可以较高（适应倾斜）
        
        # 对于极细的候选区域（如数字1），降低填充率要求
        min_fill = 0.08 if w <= 10 else 0.10
        
        if (min_w <= w <= max_w and 
            min_h <= h <= max_h and
            0.2 <= aspect_ratio <= 10.0 and  # 放宽长宽比以适应数字1和7
            bbox_area >= min_area and
            bbox_area <= img_w * img_h * 0.08 and  # 面积限制稍微放宽
            fill_ratio >= min_fill):
            
            candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'bbox_area': bbox_area,
                'center_x': x + w // 2,
                'center_y': y + h // 2,
                'fill_ratio': fill_ratio,
                'aspect_ratio': aspect_ratio
            })
    
    # 去除重叠
    candidates = remove_overlapping(candidates, overlap_thresh=0.4)  # 降低重叠阈值
    
    # 按x坐标排序
    candidates.sort(key=lambda c: c['center_x'])
    
    return candidates


def select_best_10(candidates, img_h):
    """从多于10个的候选中选择最佳的10个"""
    if len(candidates) <= 10:
        return candidates
    
    # 计算高度统计
    heights = [c['bbox'][3] for c in candidates]
    median_h = np.median(heights)
    
    # 计算每个候选的"得分"
    for c in candidates:
        h = c['bbox'][3]
        # 高度接近中位数得分高
        h_score = 1.0 - abs(h - median_h) / median_h
        # 填充率越高越好
        f_score = c['fill_ratio']
        # 面积适中得分高
        a_score = min(c['bbox_area'] / 1000, 1.0)
        
        c['score'] = h_score * 0.4 + f_score * 0.4 + a_score * 0.2
    
    # 按得分排序，取前10个
    candidates.sort(key=lambda c: c['score'], reverse=True)
    selected = candidates[:10]
    
    # 重新按x坐标排序
    selected.sort(key=lambda c: c['center_x'])
    
    return selected


def remove_overlapping(candidates, overlap_thresh=0.5):
    """去除重叠的候选区域"""
    if len(candidates) <= 1:
        return candidates
    
    # 按面积降序排序
    candidates.sort(key=lambda c: c['bbox_area'], reverse=True)
    
    keep = []
    for cand in candidates:
        x1, y1, w1, h1 = cand['bbox']
        is_overlapping = False
        
        for kept in keep:
            x2, y2, w2, h2 = kept['bbox']
            
            # 计算交集
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                min_area = min(w1 * h1, w2 * h2)
                if inter_area / min_area > overlap_thresh:
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            keep.append(cand)
    
    return keep


def recognize_with_tta(model, digit_img, device):
    """测试时增强(TTA) """
    model.eval()
    all_probs = []
    
    h, w = digit_img.shape[:2]
    
    # 精简变换：原图 + 关键旋转 + 关键缩放
    transforms_list = [('original', digit_img)]
    
    # 关键旋转角度
    for angle in [-8, 8]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(digit_img, M, (w, h), borderValue=255)
        transforms_list.append((f'rot{angle}', rotated))
    
    # 关键缩放
    for scale in [0.9, 1.1]:
        if h > 5 and w > 5:
            scaled = cv2.resize(digit_img, None, fx=scale, fy=scale)
            transforms_list.append((f'scale{scale}', scaled))
    
    # 对每种变换进行预测
    with torch.no_grad():
        for name, img in transforms_list:
            tensor = preprocess_digit_image(img).to(device)
            output = model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            all_probs.append(probs)
    
    # 平均所有预测
    avg_probs = torch.stack(all_probs).mean(dim=0)
    confidence, predicted = avg_probs.max(1)
    
    return predicted.item(), confidence.item(), avg_probs.cpu().numpy()[0]


def recognize_student_id(model, image_path, device='cpu', output_prefix=''):
    """识别学号   
    Args:
        model: 识别模型
        image_path: 图像路径
        device: 设备
        output_prefix: 输出文件前缀，用于区分不同测试图片
    """
    print("\n" + "="*50)
    print("开始识别学号")
    print("="*50)
    
    digit_images, bboxes, original_image = segment_digits_robust(image_path)
    
    if len(digit_images) == 0:
        print("错误：未检测到任何数字！")
        return "", [], []
    
    # 创建输出目录
    output_dir = 'output/exp3'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分割的数字图片
    n_digits = min(len(digit_images), 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for idx in range(10):
        if idx < n_digits:
            axes[idx].imshow(digit_images[idx], cmap='gray')
            axes[idx].set_title(f'数字 {idx+1}')
        axes[idx].axis('off')
    plt.tight_layout()
    segmented_filename = f'{output_prefix}_segmented_digits.png' if output_prefix else 'segmented_digits.png'
    plt.savefig(os.path.join(output_dir, segmented_filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分割结果已保存到：{output_dir}/{segmented_filename}")
    
    # 使用TTA识别
    predictions = []
    confidences = []
    all_probs = []
    
    for idx, digit_img in enumerate(digit_images):
        pred, conf, probs = recognize_with_tta(model, digit_img, device)
        
        predictions.append(pred)
        confidences.append(conf)
        all_probs.append(probs)
        
        top2 = np.argsort(probs)[-2:][::-1]
        print(f"数字 {idx+1}: 预测={pred} ({conf:.3f}), 次选={top2[1]} ({probs[top2[1]]:.3f})")
    
    # 简单后处理：显示低置信度警告
    if len(predictions) == 10:
        for i in range(len(predictions)):
            current_pred = predictions[i]
            current_conf = all_probs[i][current_pred]
            if current_conf < 0.5:
                top2_idx = np.argsort(all_probs[i])[-2:][::-1]
                print(f"  位置{i+1}: 低置信度 {current_pred} ({current_conf:.3f}), 次选: {top2_idx[1]} ({all_probs[i][top2_idx[1]]:.3f})")
    
    student_id = ''.join(map(str, predictions))
    avg_conf = np.mean(confidences)
    
    print(f"\n识别结果：{student_id}")
    print(f"平均置信度：{avg_conf:.4f}")
    
    # 可视化识别结果
    visualize_recognition(original_image, bboxes, predictions, confidences, student_id, output_prefix)
    
    return student_id, predictions, confidences


def visualize_recognition(image, bboxes, predictions, confidences, student_id, output_prefix=''):
    """可视化识别结果"""
    vis = image.copy()
    
    for (x, y, w, h), pred, conf in zip(bboxes, predictions, confidences):
        # 根据置信度选择颜色：绿色(高) -> 橙色(中) -> 红色(低)
        color = (0, 255, 0) if conf > 0.8 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, str(pred), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f'学号识别结果: {student_id}', fontsize=16)
    plt.axis('off')
    
    output_dir = 'output/exp3'
    result_filename = f'{output_prefix}_recognition_result.png' if output_prefix else 'recognition_result.png'
    plt.savefig(os.path.join(output_dir, result_filename), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n结果已保存到：{output_dir}/{result_filename}")


def save_results(student_id, predictions, confidences, target_id=None, output_prefix='', image_path=''):
    """保存识别结果"""
    output_dir = 'output/exp3'
    os.makedirs(output_dir, exist_ok=True)
    
    result_filename = f'{output_prefix}_result.txt' if output_prefix else 'result.txt'
    with open(os.path.join(output_dir, result_filename), 'w', encoding='utf-8') as f:
        f.write("学号识别结果\n" + "="*50 + "\n")
        if image_path:
            f.write(f"图像：{image_path}\n")
        f.write(f"识别学号：{student_id}\n")
        if target_id:
            f.write(f"目标学号：{target_id}\n")
            f.write(f"识别正确：{'是' if student_id == target_id else '否'}\n")
        f.write(f"平均置信度：{np.mean(confidences):.4f}\n")
        f.write("\n详细信息：\n")
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            f.write(f"  位置{i+1}: {pred} (置信度: {conf:.3f})\n")


def main():
    """主函数"""
    print("="*60)
    print("实验三：学号识别（深度学习方法）")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    batch_size = 64
    epochs = 10
    model_path = 'output/exp3/digit_recognition_model.pth'
    
    os.makedirs('output/exp3', exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # 创建模型
    model = CNN_Model().to(device)
    print(f"\n模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    
    # 训练或加载
    if os.path.exists(model_path):
        print(f"\n加载已保存的模型：{model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        train_losses, test_accuracies = [], []
    else:
        train_losses, test_accuracies = train_model(model, train_loader, test_loader, epochs, device)
        torch.save(model.state_dict(), model_path)
        print(f"\n模型已保存到：{model_path}")
    
    # 评估
    print("\n" + "="*50)
    test_acc = evaluate_model(model, test_loader, device)
    print(f"测试集准确率：{test_acc:.2f}%")
    print("="*50)
    
    # 识别学号
    student_id_image = 'exp3_data/1.jpg'
    target_id = '2023217499'
    
    if os.path.exists(student_id_image):
        student_id, predictions, confidences = recognize_student_id(model, student_id_image, device, 'test1')
        save_results(student_id, predictions, confidences, target_id, 'test1', student_id_image)
        
    else:
        print(f"错误：未找到图片 {student_id_image}")
    
    print("\n实验完成！")


if __name__ == '__main__':
    main()
