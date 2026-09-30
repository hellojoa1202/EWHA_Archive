# 한국인 얼굴 데이터셋을 활용한 얼굴 인식 모델 비교

| <sub>항목</sub> | <sub>내용</sub> |
|---|---|
| <sub>기간</sub> | <sub>2024-09-02 ~ 2024-12-22</sub> |
| <sub>과목</sub> | <sub>융합소프트웨어프로젝트</sub> |
| <sub>구분</sub> | <sub>개인 프로젝트</sub> |
| <sub>주제</sub> | <sub>CNN 기반 얼굴 인식 모델 성능 비교</sub> |
| <sub>데이터</sub> | <sub>한국인 얼굴 이미지 데이터셋</sub> |
| <sub>주요 모델</sub> | <sub>ResNet50, EfficientNet, MobileNetV2, InceptionV3</sub> |
| <sub>얼굴 검출</sub> | <sub>Haar-like Features, Cascade Classifier</sub> |

## 실험 범위

- <sub>한국인 얼굴 데이터 전처리</sub>
- <sub>학습·검증·테스트 데이터 분할</sub>
- <sub>CNN 모델별 학습 조건 통일</sub>
- <sub>ResNet50·EfficientNet·MobileNetV2·InceptionV3 비교</sub>
- <sub>얼굴 검출 후 인식 흐름 구성</sub>
- <sub>모델별 정확도 및 예측 결과 비교</sub>
- <sub>Colab GPU·학습 epoch 제한을 고려한 상대 성능 분석</sub>

## 파일 구성

| <sub>파일</sub> | <sub>내용</sub> |
|---|---|
| <sub>`01_data_pipeline.py`</sub> | <sub>데이터 경로 연결, 분할, 전처리, DataLoader 구성</sub> |
| <sub>`02_model_training.py`</sub> | <sub>CNN 모델 정의, 학습 함수, optimizer·scheduler 설정</sub> |
| <sub>`03_face_detection.py`</sub> | <sub>OpenCV 기반 얼굴 검출 및 검출 결과 처리</sub> |
| <sub>`04_evaluation_and_inference.py`</sub> | <sub>모델 평가, 예측 결과 시각화, 단일 이미지 추론</sub> |
| <sub>`face-recognition.py`</sub> | <sub>원본 Colab 실험 흐름 정리본</sub> |
| <sub>`data/README.md`</sub> | <sub>외부 데이터셋 및 재현 조건</sub> |

## 데이터 및 재현 조건

- <sub>외부 데이터셋: 저장소 미포함</sub>
- <sub>개인 Google Drive 경로: 저장소 미포함</sub>
- <sub>데이터 배치: `train/`, `validation/`, `test/`</sub>
- <sub>입력 크기: 224 × 224</sub>
- <sub>주요 전처리: resize, augmentation, normalization</sub>
- <sub>실행 환경: Google Colab GPU 또는 CUDA 환경</sub>
- <sub>경로 설정: `01_data_pipeline.py` 내 로컬 경로 연결 필요</sub>

## 결과

- <sub>ResNet50: 비교 모델 중 가장 높은 인식률</sub>
- <sub>얼굴 검출 후 인식: 전처리 방식에 따른 성능 변화 확인</sub>
- <sub>최종 산출물: 모델 비교 코드, 얼굴 검출 코드, 평가·추론 코드</sub>
## 자료

<sub>계획안, 중간 연구 결과 발표 자료, 얼굴 인식 모델 비교 문서와 프로젝트 코드를 함께 정리했다.</sub>
