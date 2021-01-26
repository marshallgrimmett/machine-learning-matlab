% Title: MAT 592 Assignment 1 - Linear Regression
% Author: Marshall Grimmett
% Date: 3/22/2020
% Description: This script has 2 parts. Part 1 performs linear regression
% with least squares using the normal equation. Part 2 performs linear
% regression on a new dataset containing an outlier, then compares LAD
% and LS methods.

%----------------------------------------------------------------
% Part 1
disp('Part 1')
load linreg.mat

% Compute LS using normal equation
X = [ones(1, 12).', x];
w = (X.'*X)\(X.'*y);
fprintf('Optimal w: [%.2f, %.2f]\n', w(1), w(2));

% Compute MSE
sum = 0;
for i = 1:12
    sum = sum + (y(i) - w(1) - w(2)*x(i))^2;
end
err = sum / 12;
fprintf('MSE: %.2f\n\n', err);

% Plot LS
figure(1)
scatter(x, y)
hold on
y2 = w(1) + w(2)*x;
plot(x, y2), legend('data', 'LS')
title('Part 1 - Linear Regression')
xlabel('x')
ylabel('y')
hold off


%----------------------------------------------------------------
% Part 2
disp('Part 2')
load linreg+outlier.mat

rng default % for reproducibility

% Compute LAD
fun = @(w)lad(w,x,y);
x0 = rand(2,1);
bestLAD = fminsearch(fun,x0);
fprintf('Optimal w for LAD: [%.2f, %.2f]\n', bestLAD(1), bestLAD(2));

% Compute LS
fun = @(w)ls(w,x,y);
x0 = rand(2,1);
bestLS = fminsearch(fun,x0);
fprintf('Optimal w for LS: [%.2f, %.2f]\n', bestLS(1), bestLS(2));

% Plot LAD and LS
figure(2)
scatter(x, y)
hold on
y2 = bestLAD(1) + bestLAD(2)*x;
y3 = bestLS(1) + bestLS(2)*x;
plot(x, y2, x, y3), legend('data', 'LAD', 'LS')
title('Part 2 - Linear Regression with Outlier')
xlabel('x')
ylabel('y')
hold off


function obj = lad(w,x1,y1)
obj = sum(abs(y1 - w(1) - w(2)*x1));
end

function obj = ls(w,x1,y1)
obj = sum((y1 - w(1) - w(2)*x1).^2);
end


% Output

% Part 1
% Optimal w: [3.62, 1.26]
% MSE: 0.69
% 
% Part 2
% Optimal w for LAD: [3.81, 1.20]
% Optimal w for LS: [6.05, 0.56]

