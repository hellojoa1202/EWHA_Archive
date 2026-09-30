### SNS 데이터셋을 활용한 Emotion Text Classification

<sub>---</sub>

<p><sub>웹과 텍스트마이닝개론 기말 프로젝트 · 개인</sub></p>

## Project

<p><sub>Dell 관련 SNS 텍스트를 활용해 sentiment와 emotion의 분포·관계를 분석하고, 텍스트 전처리와 네트워크 분석을 거쳐 BERT 기반 감정 분류 모델을 구현했다.</sub></p>

## Workflow

1. <sub>Dell 트윗 데이터에서 1,000개를 샘플링하고 Tweet Id, Username 등 분석에 사용하지 않는 열을 제거</sub>
2. <sub>sentiment·emotion 분포, sentiment와 emotion의 관계, 월별 변화 추이를 시각화</sub>
3. <sub>멘션과 불용어를 제거하고 단어 빈도·word cloud·treemap으로 주요 키워드 확인</sub>
4. <sub>명사 동시 출현을 기반으로 단어 네트워크를 만들고 degree·closeness·betweenness·eigenvector centrality를 계산</sub>
5. <sub>LDA로 주제를 추출하고, BERT tokenizer와 pretrained BERT를 활용한 emotion classification 모델 학습</sub>
6. <sub>confusion matrix와 classification report로 테스트 결과를 평가</sub>

## Results

- <sub>감정별 단어 분포와 sentiment·emotion 관계를 heatmap으로 확인</sub>
- <sub>단어 네트워크에서 Dell, laptop, service, customer 등 중심 키워드의 연결 관계를 시각화</sub>
- <sub>BERT 모델은 train accuracy 0.9920, test accuracy 0.6854를 기록</sub>
- <sub>anger와 joy는 비교적 높은 F1-score를 보였고, fear와 disgust는 데이터 수와 클래스 특성의 영향으로 상대적으로 낮게 나타남</sub>

## Files

1. <sub>`emotion-classification.py`: 외부 데이터 경로와 개인 파일을 제외한 전처리·텍스트 분석·네트워크·BERT 분류 코드</sub>
2. <sub>외부 SNS 데이터와 pretrained weight는 포함하지 않음</sub>
## 자료

<sub>기말 프로젝트 제출 ZIP에는 분석 코드와 제출용 보고서 문서가 함께 포함되어 있었고, 저장소에는 재현 가능한 코드 중심으로 정리했다.</sub>
