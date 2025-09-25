function [nabla] = compute_gradient(X, w, y)
    % Take the transpose of X
    X_t = X';
    
    % Compute the gradient:
    %  1) Multiply X by w to get the prediction
    %  2) Subtract the vector y (ground truth) from the prediction
    %  3) Multiply the result by the transpose of X
    %  4) Multiply by 2 to get the desired gradient
    nabla = 2 * (X_t * (X*w - y));
end