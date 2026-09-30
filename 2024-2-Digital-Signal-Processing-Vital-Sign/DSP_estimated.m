clear all; close all; clc;

% 1. 데이터 업로드
load('dsp_adcData_vital_signs.mat');

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

% 위쪽 그래프: X_vital (거리 변화)
subplot(2, 1, 1);
time_resp = (1:length(X_resp)) * T_frame; % 시간 축 (초)
plot(time_resp, X_resp);
xlabel('Time (s)');
ylabel('Distance Change (mm)'); % 거리 변화
title('Respiration Signal');

% 아래쪽 그래프: X_vital의 차분 (속도 변화 또는 신호 차분 값)
subplot(2, 1, 2);
time_heart = (1:length(X_heart)) * T_frame; % 시간 축 (초)
plot(time_heart, X_heart);
xlabel('Time (s)');
ylabel('Difference Signal'); % 차분 신호
title('Heart Signal');

% 출력된 호흡 및 심박수
fprintf('Estimated Respiration Rate: %.3f Hz\n', resp_rate);
fprintf('Estimated Heart Rate: %.3f Hz\n', heart_rate);

