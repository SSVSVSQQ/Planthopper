#YOLOv11
import yaml
import torch
from ultralytics import YOLO  # YOLOv10 检测头
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import torch.nn as nn # 如果pytorch安装成功即可导入
torch.cuda.empty_cache()

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.image_paths = []
        with open(txt_file, 'r') as file:
            for line in file:
                self.image_paths.append(line.strip())
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

# Albumentations 数据增强
transform = A.Compose([
    A.Blur(p=0.01, blur_limit=(3, 7)),
    A.MedianBlur(p=0.01, blur_limit=(3, 7)),
    A.ToGray(p=0.01),
    A.CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
    A.Resize(640, 640),
    ToTensorV2()
])

train_dataset = CustomDataset('C:/datasets/images/train/train.txt', transform=transform)

# 确保数据加载到GPU
def collate_fn(batch):
    images = torch.stack([item for item in batch])
    return images.to(device)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

params = {
    'train': "C:/datasets/images/train/train.txt",
    'val': "C:/datasets/images/val/val.txt",
    'test': "C:/datasets/images/test/test.txt",
    "nc": 3,
    'names': {
        0: 'Sogatella furcifera',  # 白背飞虱
        1: 'Drosophila melanogaster',  # 果蝇
        2: 'Brown planthopper',  # 褐飞虱
    }
}
 
with open('C:/Users/user/YOLO_ST/Lib/site-packages/ultralytics/cfg/models/v10/BHP.yaml', 'w') as file:
    yaml.dump(params, file)

# Load a model
model = YOLO('ST_YOLOv8-p2.yaml')  # build a new model from YAML

# Move the model to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'CUDA Device Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Device Count: {torch.cuda.device_count()}')
print(device)
model.to(device)

# 训练模型
results = model.train(data='C:/Users/user/YOLO_ST/Lib/site-packages/ultralytics/cfg/models/v10/BHP.yaml', epochs=500, imgsz=640, batch=8, device=device, amp=True, cache="disk")

import os 
import pandas as pd
from ultralytics import YOLOv10

# 加载训练好的 YOLOv8 模型权重
model = YOLOv10('C:/Users/user/runs/detect/train20/weights/best.pt')  # 替换为你的 YOLOv8 权重路径

# 验证集图片文件夹
val_folder = 'C:/datasets/images/val/'  # 替换为你的 val 文件夹路径

# 用于保存预测结果的列表
results_data = []

# 遍历 val 文件夹中的所有图片
image_files = [f for f in os.listdir(val_folder) if f.endswith(('.jpg', '.JPG'))]

# 预测每张图片并保存结果
for img_file in image_files:
    img_path = os.path.join(val_folder, img_file)
    
    # 模型预测
    results = model(img_path)
    
    # 获取预测的类别和框数量
    preds = results[0]  # 获取第一张图片的结果
    
    # 获取类别和预测框的数量
    labels = preds.boxes.cls if preds.boxes is not None else []  # 预测出的类别
    num_boxes = len(labels)  # 预测出的框数量

    # 如果没有检测到任何框
    if num_boxes == 0:
        results_data.append({
            'Image': img_file,
            'Class': 'No Detections',  # 无框的情况
            'Box Count': 0
        })
    else:
        # 统计每个类别的框数量
        class_counts = {}
        
        for label in labels:
            class_name = model.names[int(label)]  # 获取类别名称
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        # 将当前图片的预测结果保存到字典中
        for class_name, count in class_counts.items():
            results_data.append({
                'Image': img_file,
                'Class': class_name,
                'Box Count': count
            })

# 将结果保存到 DataFrame
df_results = pd.DataFrame(results_data)

# 保存到 Excel 文件
output_excel_path = 'F:/毕业论文/BHP1/matrix/yolov10_predictions.xlsx'  # 输出 Excel 文件路径
df_results.to_excel(output_excel_path, index=False)

print(f"预测结果已保存至: {output_excel_path}")