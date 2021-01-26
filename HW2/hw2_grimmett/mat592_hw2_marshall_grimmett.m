% Title: MAT 592 Assignment 2
% Author: Marshall Grimmett
% Date: 3/22/2020
% Description: This script has 3 parts. Part 1 performs binary 
% classification using logistic regression. Part 2 uses K-means to
% compress RGB images. Part 3 uses PCA to reduce the dimension of
% raw face images.

%----------------------------------------------------------------
% Part 1
load mnist5k.mat

% A)
figure(1)
for i = 1:9
    subplot(3,3,i);
    Xtr_temp = reshape(Xtr(i,:),[28,28]);
    imshow(Xtr_temp);
end

% B)
[tr_acc, te_acc, obj_vals] = logit(Xtr, Ytr, Xte, Yte, 5, 1, .114);

% i)
disp(['Training Accuracy: ' num2str(tr_acc)])
disp(['Testing Accuracy: ' num2str(te_acc)])
% Output
% Training Accuracy: 0.9748
% Testing Accuracy: 0.9564

% ii)
figure(2)
plot(obj_vals, 'LineWidth',2); grid on;
title('Objective Function'); xlabel('Iteration'); ylabel('F(x)');


%----------------------------------------------------------------
% Part 2
I = imread('peppers.png');

figure(3)
subplot(2,2,1)
imshow(I)
title('Original')

for i = 1:3
    I = imread('peppers.png');
    
    % quadratic computes k as 5, 20, 100 (completely unnecessary :) )
    k = (65/2)*(i-1)^2 + (15-65/2)*(i-1) + 5;
    
    I = double(reshape(I, [384*512,3]));
    [idx,C] = kmeans(I,k,'MaxIter',500);

    % Replace pixel with nearest centroid
    for j = 1:(384*512)
        I(j,:) = C(idx(j),:);
    end

    I = uint8(reshape(I, [384,512,3]));

    subplot(2,2,i+1)
    imshow(I)
    title(['k = ' num2str(k)])
end


%----------------------------------------------------------------
% Part 3
load face.mat

mu = mean(X, 2);
X_0 = X - mu;

[U,S,V] = svd(X_0);

i = 1;
k = 350;

% Compute the PCs using SVD
X_0 = U(:, 1:k)*S(1:k, 1:k);

% Reconstruct image
Recon = (X_0(i,:))*(V(:,1:k)') + mu(i);

figure(4)
subplot(1,2,1)
imshow(reshape(X(i,:), [112,92]))
title('Original')

subplot(1,2,2)
imshow(reshape(Recon, [112,92]))
title('Reconstructed')

