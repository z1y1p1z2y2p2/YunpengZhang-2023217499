import cv2
import numpy as np
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


class BicycleDetector:
    """共享单车检测器"""
    def __init__(self, model_path='yolov8n.pt'):
        print(f"正在加载模型: {model_path}")
        self.model = YOLO(model_path)
        # COCO数据集中自行车的类别ID是1 (bicycle)
        self.bicycle_class_id = 1
        
    def detect_bicycles(self, image_path, conf_threshold=0.3):
        print(f"\n正在检测图像: {image_path}")
        # 使用YOLO模型进行检测
        results = self.model(image_path, conf=conf_threshold)
        # 提取自行车检测结果
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取类别ID
                class_id = int(box.cls[0])
                
                # 只保留自行车类别
                if class_id == self.bicycle_class_id:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    
                    detection = {
                        'bbox': xyxy,  # [x1, y1, x2, y2]
                        'confidence': conf,
                        'class_name': 'bicycle'
                    }
                    detections.append(detection)
        
        print(f"检测到 {len(detections)} 辆自行车")
        return results, detections
    
    def visualize_results(self, image_path, detections, save_path=None):
        """
        可视化检测结果
        
        Args:
            image_path: 原始图像路径
            detections: 检测结果列表
            save_path: 保存路径
        """
        # 读取图像
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 绘制检测框
        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['confidence']
            
            # 绘制矩形框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加标签
            label = f"Bicycle {i+1}: {conf:.2f}"
            cv2.putText(image_rgb, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)
        plt.title(f'共享单车检测结果 (检测到 {len(detections)} 辆)', fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        
        plt.show()
    
    def save_detection_report(self, image_name, detections, output_path):
        """
        保存检测报告
        
        Args:
            image_name: 图像名称
            detections: 检测结果
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("校园共享单车检测报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"图像名称: {image_name}\n")
            f.write(f"检测到的自行车数量: {len(detections)}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("检测详情:\n")
            f.write("-"*60 + "\n\n")
            
            for i, det in enumerate(detections, 1):
                bbox = det['bbox']
                conf = det['confidence']
                
                f.write(f"自行车 #{i}:\n")
                f.write(f"  位置 (x1, y1, x2, y2): ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})\n")
                f.write(f"  中心点: ({(bbox[0]+bbox[2])/2:.1f}, {(bbox[1]+bbox[3])/2:.1f})\n")
                f.write(f"  宽度: {bbox[2]-bbox[0]:.1f} 像素\n")
                f.write(f"  高度: {bbox[3]-bbox[1]:.1f} 像素\n")
                f.write(f"  置信度: {conf:.4f}\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("检测完成\n")
            f.write("="*60 + "\n")
        
        print(f"检测报告已保存到: {output_path}")


def main():
    """主函数"""
    
    # 创建输出目录
    output_dir = 'output/exp4'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化检测器
    detector = BicycleDetector(model_path='yolov8n.pt')
    
    # 设置输入图像路径
    input_dir = 'exp4_data'
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n找到 {len(image_files)} 张图像待检测")
    
    # 对每张图像进行检测
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        image_name = os.path.splitext(image_file)[0]
        
        # 检测自行车
        results, detections = detector.detect_bicycles(image_path, conf_threshold=0.3)
        
        # 可视化结果
        save_path = os.path.join(output_dir, f'{image_name}_result.jpg')
        detector.visualize_results(image_path, detections, save_path)
        
        # 保存检测报告
        report_path = os.path.join(output_dir, f'{image_name}_detection_result.txt')
        detector.save_detection_report(image_file, detections, report_path)
    
    print("\n" + "="*60)
    print("所有检测任务完成！")
    print("="*60)


if __name__ == '__main__':
    main()
