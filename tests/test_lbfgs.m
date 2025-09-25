load("X.mat");
load(append("y", string(s), ".mat"));
[m,n] = size(X);
X_hat = zeros(m+n,m);
X_hat(1:n,1:m) = X';
X_hat(n+1:end,1:end) = lambda*eye(m);
y_hat = zeros(m+n,1);
y_hat(1:m)=y;
epsilon = 0.00001;
limit = 50;
w=zeros(m,1);
tic
[wf] = l_bfgs(X_hat, w, y_hat, m, epsilon, limit);
t_lbfgs = toc;