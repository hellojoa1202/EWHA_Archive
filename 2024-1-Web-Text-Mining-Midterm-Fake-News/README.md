### 가짜 뉴스 데이터의 텍스트 및 의미 관계 분석

<sub>---</sub>

<p><sub>웹과 텍스트마이닝개론 중간 프로젝트 · 개인</sub></p>

## Project

<p><sub>가짜 뉴스 데이터의 주제·시점·단어 분포를 시각화하고, Word2Vec을 이용해 뉴스 본문 속 단어의 의미 관계를 분석했다.</sub></p>

## Workflow

1. <sub>True 뉴스와 Fake 뉴스 데이터를 결합하고 subject, date, text 기준으로 데이터 구조 확인</sub>
2. <sub>주제별 기사 수와 월별 기사 수를 비교해 뉴스 데이터의 분포를 시각화</sub>
3. <sub>불용어를 제거한 뒤 True·Fake 뉴스의 상위 단어와 word cloud를 비교</sub>
4. <sub>Word2Vec을 학습해 `President`, `Trump`, `American`과 의미적으로 가까운 단어를 탐색</sub>
5. <sub>단어 벡터를 저장하고 재로딩해 유사도 검색 결과를 재사용</sub>

## Results

- <sub>True·Fake 뉴스의 주제별 구성과 시간대별 분포를 비교</sub>
- <sub>불용어 제거 전후의 핵심 단어를 비교해 텍스트의 주요 맥락을 확인</sub>
- <sub>Word2Vec 기반 유사 단어 검색으로 뉴스 본문 내 의미 관계를 탐색</sub>

## Files

1. <sub>`fake-news.py`: 외부 데이터 경로와 개인 파일을 제외한 전처리·시각화·Word2Vec 코드</sub>
2. <sub>외부 뉴스 데이터와 학습 결과 파일은 포함하지 않음</sub>
## 자료

<sub>중간 프로젝트 제출 ZIP에는 분석 코드와 제출용 보고서 문서가 함께 포함되어 있었고, 저장소에는 코드 중심으로 정리했다.</sub>
