import os
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import csv
import time
from torch.serialization import add_safe_globals

# torchvision 변환 클래스를 안전한 전역으로 추가
add_safe_globals([
    'torchvision.transforms.transforms.Compose',
    'torchvision.transforms.transforms.Resize',
    'torchvision.transforms.transforms.ToTensor',
    'torchvision.transforms.transforms.Normalize'
])

class DroneClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(DroneClassifier, self).__init__()
        
        # EfficientNet-B0 백본 사용
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # 분류기 부분 수정
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class DroneImageClassifier:
    def __init__(self, model_path, config_path=None, device=None):
        """
        드론 이미지 분류기 초기화
        
        Args:
            model_path (str): 모델 파일 경로 (.pth)
            config_path (str, optional): 설정 파일 경로 (.json)
            device (str, optional): 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 설정 파일 로드
        if config_path is None:
            config_path = model_path.replace('.pth', '_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # 기본 설정
            self.config = {
                'img_size': 224,
                'class_names': ['OK', 'NG'],
                'model_type': 'EfficientNet-B0'
            }
            print(f"설정 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        
        # 변환 정의
        self.transform = transforms.Compose([
            transforms.Resize((self.config['img_size'], self.config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.model = DroneClassifier(num_classes=len(self.config['class_names']), pretrained=False)
        
        try:
            # 먼저 weights_only=True로 시도 (더 안전한 방법)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"weights_only=True로 로드 실패: {e}")
            print("weights_only=False로 시도합니다. 신뢰할 수 있는 모델인 경우에만 진행하세요.")
            
            # weights_only=False로 시도 (보안 위험이 있지만 이전 버전과의 호환성을 위해)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"모델 로드 완료: {model_path}")
        print(f"클래스: {self.config['class_names']}")
    
    def predict_image(self, image_path):
        """
        단일 이미지 분류
        
        Args:
            image_path (str): 이미지 파일 경로
            
        Returns:
            tuple: (예측 클래스 인덱스, 예측 클래스 이름, 신뢰도 점수)
        """
        # 이미지 로드 및 전처리
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 예측
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
            # 결과 해석
            pred_idx = torch.argmax(probabilities).item()
            pred_class = self.config['class_names'][pred_idx]
            confidence = probabilities[pred_idx].item() * 100
                
            return pred_idx, pred_class, confidence
            
        except Exception as e:
            print(f"이미지 분류 오류 ({image_path}): {e}")
            return None, None, None
    
    def predict_folder(self, folder_path, output_csv='results.csv'):
        """
        폴더 내 모든 이미지 분류
        
        Args:
            folder_path (str): 이미지 폴더 경로
            output_csv (str): 결과를 저장할 CSV 파일 경로
            
        Returns:
            dict: 분류 결과 (파일명: (클래스, 신뢰도))
        """
        results = {}
        image_files = []
        
        # 이미지 파일 찾기
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(os.path.join(root, file))
            
        print(f"총 {len(image_files)}개 이미지 분류 시작...")
        
        # 각 이미지 분류
        for image_path in tqdm(image_files):
            pred_idx, pred_class, confidence = self.predict_image(image_path)
            if pred_class:
                results[image_path] = (pred_class, confidence)
        
        # 결과 요약
        ok_count = sum(1 for _, (cls, _) in results.items() if cls == 'OK')
        ng_count = sum(1 for _, (cls, _) in results.items() if cls == 'NG')
        
        print(f"\n분류 결과:")
        print(f"OK: {ok_count}개 ({ok_count/len(results)*100:.1f}%)")
        print(f"NG: {ng_count}개 ({ng_count/len(results)*100:.1f}%)")
        
        # CSV 저장
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['파일명', '분류 결과', '신뢰도(%)'])
            for image_path, (pred_class, confidence) in results.items():
                writer.writerow([image_path, pred_class, f"{confidence:.2f}"])
        print(f"결과가 {output_csv}에 저장되었습니다.")
            
        return results

if __name__ == "__main__":
    # 모델 경로와 테스트 폴더 경로 설정
    model_path = 'best_drone_model.pth'
    test_folder = 'src\\drone\\second\\test'
    output_csv = 'drone_classification_results.csv'
    
    # 분류기 초기화
    classifier = DroneImageClassifier(model_path=model_path)
    
    # 지정된 폴더 내 이미지 분류
    classifier.predict_folder(test_folder, output_csv=output_csv)
