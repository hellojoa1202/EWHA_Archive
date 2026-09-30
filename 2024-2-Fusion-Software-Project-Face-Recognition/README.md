# 한국인 얼굴 데이터셋을 활용한 얼굴 인식 모델 비교

| 항목 | 내용 |
|---|---|
| 기간 | 2024-09-02 ~ 2024-12-22 |
| 과목 | 융합소프트웨어프로젝트 |
| 구분 | 개인 프로젝트 |
| 주제 | CNN 기반 얼굴 인식 모델 성능 비교 |
| 데이터 | 한국인 얼굴 이미지 데이터셋 |
| 주요 모델 | ResNet50, EfficientNet, MobileNetV2, InceptionV3 |
| 얼굴 검출 | Haar-like Features, Cascade Classifier |

## 실험 범위

- 한국인 얼굴 데이터 전처리
- 학습·검증·테스트 데이터 분할
- CNN 모델별 학습 조건 통일
- ResNet50·EfficientNet·MobileNetV2·InceptionV3 비교
- 얼굴 검출 후 인식 흐름 구성
- 모델별 정확도 및 예측 결과 비교
- Colab GPU·학습 epoch 제한을 고려한 상대 성능 분석

## 파일 구성

| 파일 | 내용 |
|---|---|
| `01_data_pipeline.py` | 데이터 경로 연결, 분할, 전처리, DataLoader 구성 |
| `02_model_training.py` | CNN 모델 정의, 학습 함수, optimizer·scheduler 설정 |
| `03_face_detection.py` | OpenCV 기반 얼굴 검출 및 검출 결과 처리 |
| `04_evaluation_and_inference.py` | 모델 평가, 예측 결과 시각화, 단일 이미지 추론 |
| `face-recognition.py` | 원본 Colab 실험 흐름 정리본 |
| `data/README.md` | 외부 데이터셋 및 재현 조건 |

## 데이터 및 재현 조건

- 외부 데이터셋: 저장소 미포함
- 개인 Google Drive 경로: 저장소 미포함
- 데이터 배치: `train/`, `validation/`, `test/`
- 입력 크기: 224 × 224
- 주요 전처리: resize, augmentation, normalization
- 실행 환경: Google Colab GPU 또는 CUDA 환경
- 경로 설정: `01_data_pipeline.py` 내 로컬 경로 연결 필요

## 결과

- ResNet50: 비교 모델 중 가장 높은 인식률
- 얼굴 검출 후 인식: 전처리 방식에 따른 성능 변화 확인
- 최종 산출물: 모델 비교 코드, 얼굴 검출 코드, 평가·추론 코드
## 자료

계획안, 중간 연구 결과 발표 자료, 얼굴 인식 모델 비교 문서와 프로젝트 코드를 함께 정리했다.
