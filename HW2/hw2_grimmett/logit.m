function [tr_acc, te_acc, obj_vals] = logit(Xtr, Ytr, Xte, Yte, gamma, lam, thresh)
x = Xtr;
y = Ytr;
n = length(y);

% Parameters
max_iter = 1000;  % maximum number of iterations

fvals = [];       % store F(x) values across iterations

% Iterate
iter = 1;
w = zeros(1, 784);
w0 = 0;
fvals(iter) = F(w, w0, x, y, n, lam);
while iter < max_iter
    iter = iter + 1;
    w = w - gamma * dFw(w, w0, x, y, n, lam);  % gradient descent
    w0 = w0 - gamma * dFw0(w, w0, x, y, n);
    fvals(iter) = F(w, w0, x, y, n, lam);     % evaluate objective function
    if (iter > 100) && (fvals(iter) < thresh) % threshold
        break
    end
end

train = sigmoid(w, w0, Xtr);
test = sigmoid(w, w0, Xte);

% convert probability to values -1 and 1
count1 = 0;
count2 = 0;
for i = 1:5000 
    if train(i) < 0.5
        train(i) = 1;
    else
        train(i) = -1;
    end
    
    if test(i) < 0.5
        test(i) = 1;
    else
        test(i) = -1;
    end
    
    if train(i) == Ytr(i)
        count1 = count1 + 1;
    end
    
    if test(i) == Yte(i)
        count2 = count2 + 1;
    end
end

tr_acc = count1/n;
te_acc = count2/n;
obj_vals = fvals;

end

% Objective function F(x) to minimize in order to solve G(x)=0
function obj = F(w, w0, x, y, n, lam)
sum = 0;
for i = 1:5000
    sum = sum + log(1+exp((-y(i)*(w*x(i, :)'+w0))));
%     sum = sum + log(1+exp((-y(i)*(w*x(i, :)'+w0)))) + lam*norm(w)^2;
end
obj = (sum + lam*norm(w)^2) / n;
% obj = sum / n;
end

% Gradient of F (partial derivatives wrt w)
function obj = dFw(w, w0, x, y, n, lam)
sum = 0;
for i = 1:5000
    sum = sum - x(i, :)*y(i)*exp((-y(i)*(w*x(i, :)'+w0)))/(1+exp((-y(i)*(w*x(i, :)'+w0))));
%     sum = sum - x(i, :)*y(i)*exp((-y(i)*(w*x(i, :)'+w0)))/(1+exp((-y(i)*(w*x(i, :)'+w0)))) + 2*lam*w;
end
obj = (sum + 2*lam*w) / n;
% obj = sum / n;
end

% Gradient of F (partial derivatives wrt w0)
function obj = dFw0(w, w0, x, y, n)
sum = 0;
for i = 1:5000
    sum = sum - y(i)*exp((-y(i)*(w*x(i, :)'+w0)))/(1+exp((-y(i)*(w*x(i, :)'+w0))));
end
obj = sum / n;
end

% Sigmoid function
function obj = sigmoid(w, w0, x)
obj = zeros(5000, 1);
for i = 1:5000
    obj(i) = 1 / (1+exp(w*x(i, :)'+w0));
end
end