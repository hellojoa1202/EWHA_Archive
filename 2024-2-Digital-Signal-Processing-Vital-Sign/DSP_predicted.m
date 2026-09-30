clear all; close all; clc;

% 1. 데이터 업로드
load('dsp_adcData_vital_signs.mat');
rng(4);

% 2. 데이터 변수 할당
[N_adcs, N_chirps, N_antennas, N_frames] = size(adcData);
T_frame = 0.05;                % Frame duration (seconds)
fs = 1 / T_frame;              % Sampling frequency (Hz)

% 3. 레이더 시스템 파라미터 (수정)
radar.f0 = 24e6;                % Start frequency (Hz)
radar.B = 200e6;                % Bandwidth (Hz)
radar.SweepTime = 12.8e-6;      % Sweep time (seconds)
c = physconst('LightSpeed');    % Speed of light (m/s)
lambda = c / radar.B;           % Wavelength (meters) - 대역폭 기준으로 수정

% 4. Range FFT (수정)
s_R = zeros(N_adcs, N_chirps, N_frames, N_antennas); % FFT 결과 저장
for frameIndex = 1:N_frames
    for chirpIndex = 1:N_chirps
        s_rx_ADC = adcData(:, chirpIndex, :, frameIndex); % N_adcs x N_antennas
        for antennaIndex = 1:N_antennas
            % 정규화를 제거한 FFT
            s_R(:, chirpIndex, frameIndex, antennaIndex) = ...
                fftshift(fft(s_rx_ADC(:, antennaIndex), N_adcs));
        end
    end
end

% 5. Max Range Bin 설정
range_profile = mean(abs(s_R), [2, 3, 4]); % Range Profile 평균
[~, max_range_bin] = max(range_profile);   % 최대 에너지를 가지는 Range Bin 선택

% 6. Phase 추출 및 Unwrap
phase_signal = zeros(N_frames, 1); % Phase Signal 저장
for frameIndex = 1:N_frames
    phase_signal(frameIndex) = angle(s_R(max_range_bin, 1, frameIndex, 1)); % Phase 추출
end
phase_unwrapped = unwrap(phase_signal); % Phase Unwrap

% 7. Difference Filter 적용
X_vital = diff(phase_unwrapped); % Phase 변화량 계산

% 8. Bandpass Filter 설계
respiration_band = designfilt('bandpassiir', 'FilterOrder', 2, ...
    'HalfPowerFrequency1', 0.1, 'HalfPowerFrequency2', 0.5, ...
    'SampleRate', 1 / T_frame);

heart_band = designfilt('bandpassiir', 'FilterOrder', 2, ...
    'HalfPowerFrequency1', 1, 'HalfPowerFrequency2', 2, ...
    'SampleRate', 1 / T_frame);

% 9. Filter 적용
X_resp = filter(respiration_band, X_vital);  % 양방향 필터링 -> 한 방향 필터링으로 변경
X_heart = filter(heart_band, X_vital);  % 양방향 필터링 -> 한 방향 필터링으로 변경

% 10. Respiration 및 Heart Rate 추정
N_fft = 1024;
resp_fft = abs(fft(X_resp, N_fft));
heart_fft = abs(fft(X_heart, N_fft));

f = (0:(N_fft/2)-1) / (N_fft * T_frame); % FFT 주파수 축
resp_fft = resp_fft(1:N_fft/2);
heart_fft = heart_fft(1:N_fft/2);

[~, resp_idx] = max(resp_fft(f >= 0.1 & f <= 0.5));
resp_rate = f(resp_idx + find(f >= 0.1, 1) - 1);

[~, heart_idx] = max(heart_fft(f >= 1 & f <= 2));
heart_rate = f(heart_idx + find(f >= 1, 1) - 1);

% 11. 딥러닝 - 데이터 준비
X_train = X_vital(1:end-1);  % X_vital의 앞부분을 훈련 데이터로 사용

% y_train을 -> 호흡률과 심박수로 구분
y_train = [resp_rate * ones(size(X_train)), heart_rate * ones(size(X_train))];

% 12. 모델 설계
inputSize = 1;  % 1차원 시퀀스
layers = [
    featureInputLayer(inputSize, 'Normalization', 'zscore') % 입력 데이터 정규화
    
    fullyConnectedLayer(64, 'Name', 'fc1')                % FC layer
    reluLayer('Name', 'relu1')                            % activation func -> relu
    dropoutLayer(0.2, 'Name', 'dropout1')                 % dropout (20%)
    
    fullyConnectedLayer(64, 'Name', 'fc2')                % FC layer2
    reluLayer('Name', 'relu2')                            % activation func2 -> relu
    dropoutLayer(0.2, 'Name', 'dropout2')                 % dropout (20%)
    
    fullyConnectedLayer(2, 'Name', 'output')              
    regressionLayer('Name', 'regression')                 % regression
];

% 13. train 설정
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% 14. train
net = trainNetwork(X_train, y_train, layers, options);

% 15. predict
predicted_rates = predict(net, X_train);

% 16. predict 결과로 신호 예측
X_resp_predicted = predicted_rates(:, 1);  % 예측된 호흡률
X_heart_predicted = predicted_rates(:, 2);  % 예측된 심박수

% 17. 그래프 출력
figure;
subplot(2, 1, 1);
plot((1:length(X_resp_predicted)) * T_frame, X_resp_predicted);
xlabel('Time (s)');
ylabel('Amplitude');
title('Predicted Respiration Signal');

subplot(2, 1, 2);
plot((1:length(X_heart_predicted)) * T_frame, X_heart_predicted);
xlabel('Time (s)');
ylabel('Amplitude');
title('Predicted Heart Signal');

% 예측된 호흡률과 심박수에서 최대값 출력
predicted_resp_rate = max(X_resp_predicted);  % 예측된 호흡률의 최대값
predicted_heart_rate = max(X_heart_predicted);  % 예측된 심박수의 최대값

fprintf('Predicted Respiration Rate: %.3f Hz\n', predicted_resp_rate);  % 최대값 출력
fprintf('Predicted Heart Rate: %.3f Hz\n', predicted_heart_rate);  % 최대값 출력


