load("X.mat");

[m,n] = size(X);
X_hat = zeros(m+n,m);
X_hat(1:n,1:m) = X';
X_hat(n+1:end,1:end) = lambda*eye(m);
y_hat = zeros(m+n,1);
y_hat(1:m)=y;
tic
[V,R]=householder(X_hat);
r=triu(R(1:m,:));
[qt_b] = qr_lin(y_hat,V);
qt_b=qt_b(1:m,1);
x=r\qt_b;
t_qr = toc;