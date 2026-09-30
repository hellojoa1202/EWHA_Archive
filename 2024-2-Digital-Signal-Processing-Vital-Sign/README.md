# FMCW 레이더 기반 Vital Sign 측정

- 기간: 2024-11-25 ~ 2024-12-20
- 과목: 디지털신호처리 및 실습
- 구분: 개인 프로젝트

## 내용

FMCW 레이더의 ADC 데이터를 이용해 사람의 호흡률과 심박수를 추정하는 신호처리 파이프라인을 구현했다.

1. Range FFT로 거리 방향의 에너지 분포를 계산하고 최대 range bin을 선택
2. 선택한 bin에서 frame별 phase를 추출한 뒤 phase unwrap과 difference filter 적용
3. 호흡 대역(0.1~0.5 Hz)과 심박 대역(1~2 Hz)을 band-pass filtering
4. FFT 스펙트럼의 최대 주파수로 호흡률과 심박수 추정

프로젝트 코드에는 외부 vital-sign 데이터 파일을 포함하지 않고 MATLAB 처리 흐름과 시뮬레이션 코드를 보관했다.
