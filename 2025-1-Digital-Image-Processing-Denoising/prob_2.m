close all;
clear all;

%% 0. ?대?吏 ?꾩쿂由?
original_path = 'C:\Users\hello\Desktop\2025-1 ?대?\?붿??몄쁺?곸쿂由?HW1\original_image.png';

orig_img = imread(original_path);
orig_img = rgb2gray(orig_img);

% salt & pepper noise 異붽?
noise_img = imnoise(orig_img, 'salt & pepper');

%% 1. Average Filter & Median filter 援ы쁽
filter_sizes = [3, 5];
[H, W] = size(noise_img);
avg_filtered = cell(1,2);
med_filtered = cell(1,2);

for idx = 1:length(filter_sizes)
    k = filter_sizes(idx);
    ref = floor(k/2);
    outH = H - 2*ref; outW = W - 2*ref;
    avg_img = zeros(outH, outW);
    med_img = zeros(outH, outW);
    for i = 1+ref : H-ref
        for j = 1+ref : W-ref
            sub = double(noise_img(i-ref:i+ref, j-ref:j+ref));
            avg_img(i-ref, j-ref) = mean(sub(:));
            med_img(i-ref, j-ref) = median(sub(:));
        end
    end
    avg_filtered{idx} = uint8(avg_img);
    med_filtered{idx} = uint8(med_img);
end

%% 2. 寃곌낵 異쒕젰
figure;
subplot(3,2,1); imshow(noise_img); title('Test Image (Noise)');
subplot(3,2,2); imshow(noise_img); title('Test Image (Noise)');
subplot(3,2,3); imshow(avg_filtered{1}); title('3x3 Average Filter');
subplot(3,2,4); imshow(med_filtered{1}); title('3x3 Median Filter');
subplot(3,2,5); imshow(avg_filtered{2}); title('5x5 Average Filter');
subplot(3,2,6); imshow(med_filtered{2}); title('5x5 Median Filter');

%% 3. ?댁옣 ?⑥닔 ?ъ슜?섏뿬 cross-check!
built_in_avg_filtered = cell(1,2);
built_in_median_filtered = cell(1,2);
for idx = 1:2
    k = filter_sizes(idx);
    built_in_avg_filtered{idx} = imfilter(noise_img, fspecial('average', k), 'replicate');
    built_in_median_filtered{idx} = medfilt2(noise_img, [k k]);
end

figure;
subplot(4,4,1);
imshow(avg_filtered{1});
title('Manual 3x3 Avg');

subplot(4,4,2);
imshow(built_in_avg_filtered{1});
title('Built-in 3x3 Avg');

subplot(4,4,3);
imshow(med_filtered{1});
title('Manual 3x3 Med');

subplot(4,4,4);
imshow(built_in_median_filtered{1});
title('Built-in 3x3 Med');

subplot(4,4,5);
imhist(avg_filtered{1});
title('Hist: M 3x3 Avg');

subplot(4,4,6);
imhist(built_in_avg_filtered{1});
title('Hist: B 3x3 Avg');

subplot(4,4,7);
imhist(med_filtered{1});
title('Hist: M 3x3 Med');

subplot(4,4,8);
imhist(built_in_median_filtered{1});
title('Hist: B 3x3 Med');

subplot(4,4,9);
imshow(avg_filtered{2});
title('Manual 5x5 Avg');

subplot(4,4,10);
imshow(built_in_avg_filtered{2});
title('Built-in 5x5 Avg');

subplot(4,4,11);
imshow(med_filtered{2});
title('Manual 5x5 Med');

subplot(4,4,12);
imshow(built_in_median_filtered{2});
title('Built-in 5x5 Med');

subplot(4,4,13);
imhist(avg_filtered{2});
title('Hist: M 5x5 Avg');

subplot(4,4,14);
imhist(built_in_avg_filtered{2});
title('Hist: B 5x5 Avg');

subplot(4,4,15);
imhist(med_filtered{2});
title('Hist: M 5x5 Med');

subplot(4,4,16);
imhist(built_in_median_filtered{2});
title('Hist: B 5x5 Med');
