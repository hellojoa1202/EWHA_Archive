close all
clear all

%% 0. ?대?吏 ?낅줈??& 洹몃젅?댁뒪耳??蹂??
original_path = 'C:\Users\hello\Desktop\2025-1 ?대?\?붿??몄쁺?곸쿂由?HW1\original_image.png';
target_path = 'C:\Users\hello\Desktop\2025-1 ?대?\?붿??몄쁺?곸쿂由?HW1\target_image.png';

orig_img = imread(original_path);
orig_img = rgb2gray(orig_img);
target_img = imread(target_path);
target_img = rgb2gray(target_img);

%% 1. Histogram Equalization
% 0. ?대?吏 ?ㅼ젙
[rows, cols] = size(orig_img);
num_pixels = rows * cols;
L = 256;     %intensity level (0~255)
hist_orig = zeros(1, L);

% 1. ?먮낯 image histogram ?앹꽦 諛?異쒕젰
for i = 1:rows
    for j = 1:cols
        intensity = orig_img(i,j); % 0~255
        hist_orig(intensity+1) = hist_orig(intensity+1) + 1;
    end
end

figure;
subplot(3,2,1);
imshow(orig_img);
title('Original Image');

subplot(3,2,2);
bar(0:L-1, hist_orig);
title('Original Image Histogram');

% 2. PDF & CDF 怨꾩궛
% PDF : p_r(r_k) = h(r_k) / (M횞N)
pdf_orig = hist_orig / num_pixels;
% CDF : T(r_k) = (L??)횞?묅굧瘦쇄굦??롡겱k??p_r(r_j)
cdf_orig = zeros(1, L);
cdf_orig(1) = pdf_orig(1);
for k = 2:L
    cdf_orig(k) = cdf_orig(k-1) + pdf_orig(k);
end

% 3. Transformation function
T = round((L-1) * cdf_orig);

% 4. 媛?pixel??mapping(T) ?곸슜 -> histogram equalized ?대?吏 ?앹꽦
equalized_img = zeros(size(orig_img));
for i = 1:rows
    for j = 1:cols
        equalized_img(i,j) = T(orig_img(i,j) + 1);
    end
end
% 8鍮꾪듃 ?뺤닔?뺤쑝濡?蹂???꾩슂 (?닿굅 ?놁쑝硫?0~1濡??몄떇?댁꽌 ?곗깋?쇰줈 ?섏샂)
equalized_img = uint8(equalized_img);

% 5. histogram 怨꾩궛
hist_eq = zeros(1, L);
for i = 1:rows
    for j = 1:cols
        intensity = equalized_img(i,j);
        hist_eq(intensity+1) = hist_eq(intensity+1) + 1;
    end
end

% 6. 寃곌낵 異쒕젰
subplot(3,2,3);
imshow(equalized_img);
title('Histogram Equalized Image');

subplot(3,2,4);
bar(0:L-1, hist_eq);
title('Equalized Image Histogram');

%% 2. Histogram Matching
% 1. target image histogram ?앹꽦
[rows_t, cols_t] = size(target_img);
total_pixels_t = rows_t * cols_t;
hist_target = zeros(1, L);
for i = 1:rows_t
    for j = 1:cols_t
        intensity = target_img(i,j);
        hist_target(intensity+1) = hist_target(intensity+1) + 1;
    end
end

% 2. target PDF & CDF 怨꾩궛
pdf_target = hist_target / total_pixels_t;
cdf_target = zeros(1, L);
cdf_target(1) = pdf_target(1);
for k = 2:L
    cdf_target(k) = cdf_target(k-1) + pdf_target(k);
end

% 3. target image Transformation function
G = round((L-1) * cdf_target);

% 4. equalized 媛?s??????寃잛쓽 G? 媛??洹쇱젒??z媛?李얘린
% -> 媛??쎌??????留ㅼ묶 ?섑뻾!
matched_img = zeros(size(orig_img));
for i = 1:rows
    for j = 1:cols
        s_val = equalized_img(i,j);
        diff = abs(G - double(s_val));
        [~, idx] = min(diff);
        matched_img(i,j) = idx - 1; % MATLAB ?몃뜳??蹂댁젙
    end
end
matched_img = uint8(matched_img);

% 5. matched ?대?吏???덉뒪?좉렇??怨꾩궛
hist_matched = zeros(1, L);
for i = 1:rows
    for j = 1:cols
        intensity = matched_img(i,j);
        hist_matched(intensity+1) = hist_matched(intensity+1) + 1;
    end
end

% 6. 寃곌낵 異쒕젰
subplot(3,2,5);
imshow(matched_img);
title('Histogram Matched Image');

subplot(3,2,6);
bar(0:L-1, hist_matched);
title('Matched Histogram');

%% 3. ?댁옣 ?⑥닔 ?ъ슜?섏뿬 cross-check!
equalized_builtin = histeq(orig_img);
[h_builtin_eq, x_builtin_eq] = imhist(equalized_builtin);
matched_builtin = imhistmatch(orig_img, target_img);
[h_builtin_match, x_builtin_match] = imhist(matched_builtin);

figure;
subplot(2,4,1);
imshow(equalized_img);
title('Manual Equalized Image');

subplot(2,4,2);
bar(0:L-1, hist_eq);
title('Manual Equalized Histogram');

subplot(2,4,3);
imshow(matched_img);
title('Manual Matched Image');

subplot(2,4,4);
bar(0:L-1, hist_matched);
title('Manual Matched Histogram');

subplot(2,4,5);
imshow(equalized_builtin);
title('Built-in Equalized Image');

subplot(2,4,6);
bar(x_builtin_eq, h_builtin_eq);
title('Built-in Equalized Histogram');

subplot(2,4,7);
imshow(matched_builtin);
title('Built-in Matched Image');

subplot(2,4,8);
bar(x_builtin_match, h_builtin_match);
title('Built-in Matched Histogram');
