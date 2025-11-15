import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class DroneImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # OK 폴더의 이미지들 (라벨 0)
        ok_path = os.path.join(root_dir, 'OK')
        if os.path.exists(ok_path):
            self._find_images(ok_path, label=0)
        
        # NG 폴더의 이미지들 (라벨 1)
        ng_path = os.path.join(root_dir, 'NG')
        if os.path.exists(ng_path):
            self._find_images(ng_path, label=1)
            
        print(f"총 이미지 개수: {len(self.images)}")
        print(f"OK 이미지: {self.labels.count(0)}, NG 이미지: {self.labels.count(1)}")
    
    def _find_images(self, folder_path, label):
        """폴더 내의 모든 이미지를 재귀적으로 찾는 함수"""
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # 폴더인 경우 재귀적으로 탐색
            if os.path.isdir(item_path):
                self._find_images(item_path, label)
            # 파일인 경우 이미지인지 확인
            elif os.path.isfile(item_path) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.images.append(item_path)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 에러: {e}")
            # 기본 이미지 생성
            image = Image.new('RGB', (224, 224), color='white')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# AugmentedDataset 클래스를 전역으로 이동
class AugmentedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image_path = self.subset.dataset.images[self.subset.indices[idx]]
        label = self.subset.dataset.labels[self.subset.indices[idx]]
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 에러: {e}")
            # 기본 이미지 생성
            image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                image = self.transform(image)
        
        return image, label

class DroneClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
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

class DroneModelTrainer:
    def __init__(self, data_path, img_size=224, batch_size=32, device=None):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"사용 디바이스: {self.device}")
        
        # 데이터 변환 정의
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def prepare_data(self, val_split=0.2):
        """데이터 준비 및 분할"""
        # 전체 데이터셋 로드
        full_dataset = DroneImageDataset(self.data_path, transform=self.val_transform)
        
        if len(full_dataset) == 0:
            raise ValueError("데이터셋이 비어있습니다. 경로를 확인해주세요.")
        
        # 훈련/검증 분할
        dataset_size = len(full_dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # 전역으로 이동된 AugmentedDataset 클래스 사용
        train_dataset_aug = AugmentedDataset(train_dataset, self.train_transform)
        val_dataset_clean = AugmentedDataset(val_dataset, self.val_transform)
        
        # 데이터 로더 생성
        self.train_loader = DataLoader(
            train_dataset_aug, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset_clean, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        print(f"훈련 샘플 수: {len(train_dataset)}")
        print(f"검증 샘플 수: {len(val_dataset)}")
        
        # 클래스 분포 확인
        ok_count = sum(1 for _, label in full_dataset if label == 0)
        ng_count = sum(1 for _, label in full_dataset if label == 1)
        print(f"OK 샘플: {ok_count}, NG 샘플: {ng_count}")
        
    def build_model(self):
        """모델 생성"""
        self.model = DroneClassifier(num_classes=2, pretrained=True)
        self.model.to(self.device)
        
        # 모델 파라미터 수 출력
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"총 파라미터: {total_params:,}")
        print(f"훈련 가능한 파라미터: {trainable_params:,}")
        
    def train_model(self, epochs=50, learning_rate=0.001):
        """모델 훈련"""
        # 클래스 가중치 계산 (불균형 데이터 처리)
        ok_count = sum(1 for _, label in self.train_loader.dataset if label == 0)
        ng_count = sum(1 for _, label in self.train_loader.dataset if label == 1)
        total = ok_count + ng_count
        
        class_weights = torch.tensor([total/(2*ok_count), total/(2*ng_count)]).to(self.device)
        
        # 손실 함수와 옵티마이저
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        
        best_val_acc = 0.0
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            # 훈련 단계
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # 검증 단계
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # 평균 계산
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(self.val_loader)
            val_acc = 100. * val_correct / val_total
            
            # 히스토리 저장
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 60)
            
            # 학습률 스케줄러 업데이트
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_acc)
            curr_lr = optimizer.param_groups[0]['lr']
            
            # 학습률이 변경되었는지 확인하고 출력
            if curr_lr != prev_lr:
                print(f'학습률이 {prev_lr:.6f}에서 {curr_lr:.6f}로 변경되었습니다.')
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model('best_drone_model.pth')
                print(f'새로운 최고 성능! 검증 정확도: {val_acc:.2f}%')
            else:
                patience_counter += 1
                
            # 조기 종료
            if patience_counter >= patience:
                print(f'조기 종료! {patience} 에포크 동안 개선되지 않음')
                break
        
        print(f'훈련 완료! 최고 검증 정확도: {best_val_acc:.2f}%')
        
    def save_model(self, filepath):
        """모델 저장"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'img_size': self.img_size,
            'class_names': ['OK', 'NG'],
            'transform': self.val_transform
        }
        torch.save(save_dict, filepath)
        
        # 설정 정보도 JSON으로 저장
        config = {
            'img_size': self.img_size,
            'class_names': ['OK', 'NG'],
            'model_type': 'EfficientNet-B0'
        }
        
        config_path = filepath.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f'모델 저장 완료: {filepath}')
        print(f'설정 저장 완료: {config_path}')
        
    def plot_training_history(self):
        """훈련 과정 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 정확도
        axes[0].plot(self.history['train_acc'], label='Training Accuracy', marker='o')
        axes[0].plot(self.history['val_acc'], label='Validation Accuracy', marker='s')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True)
        
        # 손실
        axes[1].plot(self.history['train_loss'], label='Training Loss', marker='o')
        axes[1].plot(self.history['val_loss'], label='Validation Loss', marker='s')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate_model(self):
        """모델 평가"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='평가 중'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # 분류 보고서
        class_names = ['OK', 'NG']
        print("\n=== 분류 보고서 ===")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # 혼동 행렬
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

# 사용 예시
if __name__ == "__main__":
    # 데이터 경로 설정
    data_path = r"src\drone\second"
    
    # 트레이너 초기화
    trainer = DroneModelTrainer(
        data_path=data_path,
        img_size=224,
        batch_size=32
    )
    
    # 데이터 준비
    trainer.prepare_data(val_split=0.2)
    
    # 모델 구축
    trainer.build_model()
    
    # 모델 훈련
    trainer.train_model(epochs=50, learning_rate=0.001)
    
    # 훈련 과정 시각화
    trainer.plot_training_history()
    
    # 모델 평가
    trainer.evaluate_model()
    
    print("훈련 완료! 'best_drone_model.pth' 파일이 저장되었습니다.")
    print("이 모델을 다른 폴더에서 분류기로 사용할 수 있습니다.")
