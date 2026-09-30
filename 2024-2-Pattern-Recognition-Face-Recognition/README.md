### 한국인 얼굴 데이터셋을 활용한 얼굴 인식 모델 비교

---

<p><sub>ASD 아동의 감정 이해를 돕기 위한 얼굴 표정 분류 모델 설계·비교 프로젝트</sub></p>

## Project

<p><sub>6가지 감정 클래스(anger, disgust, fear, happy, pain, sad)로 구성된 이미지 데이터셋을 활용해 얼굴 표정 분류 모델을 설계하고, CNN 구조·Loss Function·Optimizer 조합에 따른 성능을 비교했다.</sub></p>

## Workflow

1. <sub>데이터셋 확인: 1,200개 RGB 이미지와 6개 감정 클래스의 분포를 확인</sub>
2. <sub>전처리: dlib 얼굴 검출 및 크롭, 224×224 리사이즈, 좌우 반전·회전·affine·색상 변화·grayscale 변환 적용</sub>
3. <sub>데이터 분할: train · validation · test를 6:2:2로 구성하고 DataLoader로 미니배치 학습</sub>
4. <sub>모델 설계: 사전 학습 가중치를 활용하고 마지막 출력 계층을 6개 클래스로 수정</sub>
5. <sub>비교 실험: CNN 모델, Loss Function, Optimizer를 각각 비교한 뒤 최종 조합을 선정</sub>

## Model Comparison

<table>
  <thead>
    <tr>
      <th><sub>모델</sub></th>
      <th><sub>Best validation accuracy</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr><td><sub>ResNet50</sub></td><td><sub>0.7173</sub></td></tr>
    <tr><td><sub>LeNet5</sub></td><td><sub>0.2827</sub></td></tr>
    <tr><td><sub>AlexNet</sub></td><td><sub>0.6624</sub></td></tr>
    <tr><td><sub>VGGNet</sub></td><td><sub>0.7131</sub></td></tr>
    <tr><td><sub>SENet</sub></td><td><sub>0.6034</sub></td></tr>
    <tr><td><sub>EfficientNet</sub></td><td><sub>0.5232</sub></td></tr>
    <tr><td><sub>DenseNet</sub></td><td><sub>0.6371</sub></td></tr>
    <tr><td><sub>ResNet152</sub></td><td><sub>0.7426</sub></td></tr>
  </tbody>
</table>

<p><sub>30 epochs 기준 ResNet152, ResNet50, VGGNet 순으로 높은 검증 정확도를 보였다.</sub></p>

## Loss Function and Optimizer

<table>
  <thead>
    <tr>
      <th><sub>비교 항목</sub></th>
      <th><sub>Best validation accuracy</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr><td><sub>CrossEntropyLoss</sub></td><td><sub>0.7722</sub></td></tr>
    <tr><td><sub>MSELoss</sub></td><td><sub>0.6962</sub></td></tr>
    <tr><td><sub>MultiLabelSoftMarginLoss</sub></td><td><sub>0.7511</sub></td></tr>
    <tr><td><sub>MultiMarginLoss</sub></td><td><sub>0.7553</sub></td></tr>
    <tr><td><sub>RMSprop</sub></td><td><sub>0.7553</sub></td></tr>
  </tbody>
</table>

<p><sub>Loss Function 비교에서는 CrossEntropyLoss, Optimizer 비교에서는 RMSprop의 검증 정확도가 가장 높았다. 최종 모델은 ResNet152 + CrossEntropyLoss + RMSprop 조합으로 구성했다.</sub></p>

## Analysis

- <sub>모델·Loss Function·Optimizer의 조합에 따라 정확도의 상대적인 순위가 달라질 수 있어 비교군 외 조건을 통제했다.</sub>
- <sub>LeNet5는 데이터셋의 클래스 수와 변형 범위에 비해 구조가 단순해 상대적으로 낮은 성능을 보였다.</sub>
- <sub>Adagrad와 SGD는 동일한 낮은 learning rate에서 업데이트가 느려 성능이 낮게 나타난 것으로 분석했다.</sub>
- <sub>외부 이미지에서는 happy·disgust 예측이 비교적 잘 이루어졌지만 anger·sad처럼 표정 차이가 작은 클래스에서는 오분류가 발생했다.</sub>
- <sub>추가 데이터와 클래스별 충분한 샘플, 멀티모달 정보 결합을 통해 성능을 보완할 수 있다.</sub>

## Files

1. <sub>`face-recognition.py`: 외부 데이터와 개인 경로를 제외한 전처리·모델 비교·평가 코드</sub>
2. <sub>외부 데이터셋과 얼굴 랜드마크 파일은 포함하지 않음</sub>
## 자료

Assignment 1~3 코드·보고서, 데이터 소개서, 모델 비교 자료와 최종 얼굴 인식 프로젝트 제출본을 별도로 보관했다.
