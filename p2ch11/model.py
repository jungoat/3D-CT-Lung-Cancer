import math

from torch import nn as nn


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8): # 입력 채널 1, 첫 conv layer의 출력 채널 수 8
        super().__init__() # 부모클래스인 nn.Module의 초기화 함수

# 테일 : 입력값을 이동시키고 비율을 조정해서 정규화함.
        self.tail_batchnorm = nn.BatchNorm3d(1) # (B, 1, D, H, W)

#  백본: 특징 추출
        self.block1 = LunaBlock(in_channels, conv_channels) # 입력 채널 1 -> 8
        self.block2 = LunaBlock(conv_channels, conv_channels * 2) # 채널 8 -> 16
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4) # 채널 16 -> 32
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8) # 채널 32 -> 64

# 헤드 : 분류.  1152는 flatten된 feature vector의 길이.
        self.head_linear = nn.Linear(1152, 2) # block4의 출력 텐서를 벡터로 펼쳐서 2개의 클래스(결절/비결절) 분류
        self.head_softmax = nn.Softmax(dim=1) # 각 행마다 softmax함.

        self._init_weights() # 가중치 초기화

    def _init_weights(self): # 모델 내부 레이어들을 순회하며 가중치와 bias를 직접 초기화.
        for m in self.modules(): # 모델 내 모든 하위 레이어를 하나씩 반환함.
            if type(m) in { # 현재 레이어 타입이 아래 5자지중 하나라면 초기화한다.
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_( # kaiming normal 방식으로 초기화
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None: # bias가 존재할 경우 초기화
# fan_out : 출력 채널 수 x 커널 사이즈 -> 출력 방향으로 몇 개의 뉴런에 연결되는지
# bound : 정규분포 범위 설정용 -> bias 초기화 범위 결정
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound) # bias를 정규분포로 초기화.


    def forward(self, input_batch): # input_batch : [batch_size, 1, D, H, W] 크기의 3D CT 입력
# forward는 model(input_batch) 호출 시 자동 실행되는 함수.
        
        bn_output = self.tail_batchnorm(input_batch) # 입력을 정규화함.

# 각 블록을 통과하면서 고차원 피처 추출
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

# Flatten 펼치기
# view(..., -1)은 flatten 연산임. batch_size는 그대로 두고 나머지를 모두 1차원으로 펼침.
# 결과는 [batch_size, feature_dim] 형태의 텐서로 변환됨.
        conv_flat = block_out.view(
            block_out.size(0),
            -1,
        )
        linear_output = self.head_linear(conv_flat) # 특징 벡터를 받아서 2개의 클래스로 분류함.

# 둘 다 리턴해서 필요에 따라 선택적으로 사용 가능함.
        return linear_output, self.head_softmax(linear_output) # raw logits(모델의 출력점수), softmax확률(총합 1, 각 클래스 확률)


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

# 입력 채널 수 -> 출력 채널 수로 3D 합성곱
# 커널 : 3x3x3 필터, padding=1: 출력크기를 입력과 동일하게 유지, bias=True:bias파라미터 포함.

        self.conv1 = nn.Conv3d(
            in_channels, conv_channels, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU(inplace=True) # ReLU 활성화 함수 적용, inplace=True:메모리 사용을 줄이기 위해 입력을 바로 수정.
        self.conv2 = nn.Conv3d( # 합성곱 한번 더 수행함.
            conv_channels, conv_channels, kernel_size=3, padding=1, bias=True # 이번에는 입력 채널, 출력 채널 같음.
        )
        self.relu2 = nn.ReLU(inplace=True)

# 입력의 크기를 절반으로 줄임. kernel_size=2, stride=2. 2x2x2 공간에서 가장 큰 값만 남김.
        self.maxpool = nn.MaxPool3d(2, 2) 

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)

        return self.maxpool(block_out)