import torch
import time

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ===========================================================
# 0. 실행 장치 설정
# ===========================================================

# GPU를 사용할 수 있으면 CUDA, 아니면 CPU 사용
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)

if torch.cuda.is_available():
    print("사용 GPU:", torch.cuda.get_device_name(0))


# 결과를 어느 정도 동일하게 재현하기 위한 seed
torch.manual_seed(11)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(11)


# ===========================================================
# 1. 데이터
# ===========================================================

BATCH_SIZE = 128

# CIFAR-10의 클래스 이름
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ===========================================================
# 1-2. 데이터 전처리
# ===========================================================

# transforms.Compose를 통해 데이터들을 일괄적으로 전처리(transform)할 수 있다. 이거 없으면 이미지마다 전부 따로 따로 해야 함.
transform = transforms.Compose([
    # 이미지 혹은 Numpy배열 → PyTorch "Tensor" 즉 텐서 변환이다.
    # 픽셀값도 0~255에서 0~1로 변환
    # 채널 순서가 이것을 안하면, Height, Width, Channel 순서라서 이걸 해야 Channel, Height, Width 가 된다. 
    transforms.ToTensor(), 

    # RGB 각 채널을 대략 -1~1 범위로 정규화 (표준화)
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


# CIFAR-10 훈련 데이터 50,000장
full_train_dataset = datasets.CIFAR10(
    root="./_data",
    train=True,
    download=False,
    transform=transform #전처리 방식을 내가 정한 전처리 방식으로 넣을 수 있다. transform이 전처리 옵션이다.
)

# CIFAR-10 테스트 데이터 10,000장
test_dataset = datasets.CIFAR10(
    root="./_data",
    train=False,
    download=False,
    transform=transform
)


# 훈련 데이터 50,000장을
# train 40,000장, validation 10,000장으로 분리
train_dataset, val_dataset = random_split(
    full_train_dataset, #full_train_dataset[0] 은 (이미지 한 장 + 정답 라벨 하나 --> 구성이 이렇게 되어 있음.) 가져오는 것이다.
    lengths=[40000, 10000],
    generator=torch.Generator().manual_seed(11)
)


# Dataset을 "배치 단위"로 꺼내 주는 DataLoader
""" 

Dataset: 이미지 40,000장 보관
               ↓ DataLoader
1번째 배치: 128장
2번째 배치: 128장
3번째 배치: 128장
...

"""
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

""" 
Dataset: 데이터 한 건을 어떻게 가져올 것인가
DataLoader: 데이터 여러 건을 어떻게 묶어서(배치단위) 공급할 것인가?
"""

print("Train 데이터 수:", len(train_dataset))
print("Validation 데이터 수:", len(val_dataset))
print("Test 데이터 수:", len(test_dataset))


# 배치 하나를 꺼내서 모양 확인
sample_images, sample_labels = next(iter(train_loader))

print("이미지 배치 shape:", sample_images.shape)
print("라벨 배치 shape:", sample_labels.shape)
#exit()

# ===========================================================
# 2. 모델 구성
# ===========================================================

class BasicCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 이미지 특징 추출
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # 분류
        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 32×32 → Pooling 두 번 → 8×8
            nn.Linear(
                in_features=64 * 8 * 8,
                out_features=128
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # CIFAR-10 클래스 10개
            nn.Linear(
                in_features=128,
                out_features=10
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# 모델을 GPU 또는 CPU로 이동
model = BasicCNN().to(device)

print(model)


# ===========================================================
# 3. 손실함수와 Optimizer
# ===========================================================

# Keras의 sparse_categorical_crossentropy에 대응
criterion = nn.CrossEntropyLoss()

# Keras의 optimizer="adam"에 대응
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ===========================================================
# Validation/Test 평가 함수
# ===========================================================

def evaluate(model, data_loader, criterion):

    # 평가 모드
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # 평가에서는 기울기를 계산하지 않음
    with torch.no_grad():

        for images, labels in data_loader:

            # 데이터도 모델과 같은 GPU 또는 CPU로 이동
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_count += labels.size(0)

    average_loss = total_loss / total_count
    accuracy = total_correct / total_count

    return average_loss, accuracy


# ===========================================================
# 3-2. 훈련
# ===========================================================

EPOCHS = 10

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

start_time = time.time()

for epoch in range(EPOCHS):

    # 훈련 모드
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_count = 0

    for images, labels in train_loader:

        # 이미지와 라벨을 GPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        # 이전 배치에서 계산한 기울기 초기화
        optimizer.zero_grad()

        # 순전파
        outputs = model(images)

        # 손실 계산
        loss = criterion(outputs, labels)

        # 역전파
        loss.backward()

        # 가중치 업데이트
        optimizer.step()

        train_loss_sum += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_count += labels.size(0)

    train_loss = train_loss_sum / train_count
    train_accuracy = train_correct / train_count

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion
    )

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

end_time = time.time()

print(
    "훈련 시간:",
    round(end_time - start_time, 2),
    "초"
)


# ===========================================================
# 4. 평가
# ===========================================================

test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion
)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))


# ===========================================================
# 4-2. 예측
# ===========================================================

# 테스트 데이터에서 배치 하나 가져오기
test_images, test_labels = next(
    iter(test_loader)
)

test_images = test_images.to(device)

model.eval()

with torch.no_grad():
    test_outputs = model(test_images)

    predicted_classes = torch.argmax(
        test_outputs,
        dim=1
    )


# GPU Tensor을 다시 CPU로 이동
predicted_classes = predicted_classes.cpu()

print("\n실제값 / 예측값")

for i in range(10):
    actual_name = class_names[
        test_labels[i].item()
    ]

    predicted_name = class_names[
        predicted_classes[i].item()
    ]

    print(
        f"{i + 1}번 | "
        f"실제: {actual_name:10s} | "
        f"예측: {predicted_name}"
    )