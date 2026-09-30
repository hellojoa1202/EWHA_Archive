% 1. 두 파일에 있는 그래프와 rate 데이터 값 로드
evalc('run(''DSP_estimated.m'');'); % DSP_estimated.m 변수 로드
evalc('run(''DSP_predicted.m'');'); % DSP_predicted.m 변수 로드

% 3. 두 파일에 있는 rate 값을 각각 출력
fprintf('Estimated Respiration Rate: %.3f Hz\n', resp_rate);
fprintf('Predicted Respiration Rate: %.3f Hz\n', predicted_resp_rate);
fprintf('---------------------------------\n');
fprintf('Estimated Heart Rate: %.3f Hz\n', heart_rate);
fprintf('Predicted Heart Rate: %.3f Hz\n', predicted_heart_rate);

% 4. 상대 오차 (Relative Error) 계산
% 상대 오차 = (예측값 - 실제값) / 실제값 * 100

resp_rate_relative_error = abs((predicted_resp_rate - resp_rate) / resp_rate) * 100;
heart_rate_relative_error = abs((predicted_heart_rate - heart_rate) / heart_rate) * 100;

% 결과 출력
fprintf('---------------------------------\n');
fprintf('Respiration Rate Relative Error: %.2f%%\n', resp_rate_relative_error);
fprintf('Heart Rate Relative Error: %.2f%%\n', heart_rate_relative_error);
