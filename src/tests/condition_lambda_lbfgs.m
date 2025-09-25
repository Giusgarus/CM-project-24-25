addpath('../');
load("X.mat");

[m,n] = size(X);
X_hat = zeros(m+n,m);
X_hat(1:n,1:m) = X';

time = zeros(1,10);
for k=1:10
    s = k;
    e = 0;
    lambda = 0.0001;
    for i=1:9
        X_hat(n+1:end,:) = lambda*eye(m);
        for j=1:10
            test_lbfgs;
            e = e + t_lbfgs;
            time(1,j)=t_lbfgs;
        end
        v = var(time);
        e = e/10;
        row = [lambda, e, v, epsilon, limit];
        wf = wf';
        row_results = [wf(1,1:5)];
        save(append("execution_time_LBFGS_",string(s),".txt"), "row","-ascii", "-append");
        save("results_lbfgs.txt", "row_results", "-ascii");
        lambda = lambda * 10;
    end
end
