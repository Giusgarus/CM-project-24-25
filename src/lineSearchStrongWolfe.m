function [alpha, w_new] = lineSearchStrongWolfe(X, y, w, p, g, c1, c2)
    % lineSearchStrongWolfe executes a line search that satisfies:
    %  1) Armijo (sufficient decrease)
    %  2) Strong Wolfe conditions
    % for an example cost function f(w) = 0.5 * norm(Xw - y)^2
    %
    % Inputs:
    %   - X: Data matrix (n x d)
    %   - y: Target vector (n x 1)
    %   - w: Current weight (parameter) vector (d x 1)
    %   - p: Descent direction (d x 1)
    %   - g: Current gradient at w (d x 1)
    %   - c1: Armijo parameter (typically small, e.g. 1e-4)
    %   - c2: Strong Wolfe parameter (~0.9, with c1 < c2 < 1)
    %
    % Outputs:
    %   - alpha: Found step size
    %   - w_new: Updated weight vector, w + alpha * p
    
    % ---- Example objective function (least squares) -------------
    % f(w)    = 0.5 * (Xw - y)'(Xw - y)
    % grad(w) = X' (Xw - y)
    function [fval, gradw] = funObj(wCurrent)
        r     = X * wCurrent - y;  % residual
        fval  = 0.5 * (r' * r);
        gradw = X' * r;
    end

    % Evaluate the objective and gradient dot product at the starting point
    [f0, ~] = funObj(w);
    gp = g' * p;  % Dot product between gradient and direction

    % If p is not a descent direction (gp >= 0),
    % force p = -g to avoid potential issues.
    if gp >= 0
        p  = -g;
        gp = g' * p;
    end

    % Set initial step size and maximum number of attempts
    alpha   = 1.0;  % initial step
    maxIter = 20;   % maximum number of reductions

    for i = 1:maxIter
        
        % Evaluate the objective at w + alpha*p
        w_try           = w + alpha * p;
        [fAlpha, gTry]  = funObj(w_try);
        gpt             = gTry' * p;  % gradient dot product at new point
        
        % 1) Armijo condition (sufficient decrease):
        %    f(w + alpha*p) <= f(w) + c1 * alpha * g' * p
        if fAlpha > f0 + c1 * alpha * gp
            % If not satisfied, reduce alpha
            alpha = alpha / 2;
            continue;
        end
        
        % 2) Strong Wolfe condition:
        %    |g(w + alpha*p)' * p| <= c2 * |g' * p|
        if abs(gpt) > c2 * abs(gp)
            % If not satisfied, reduce alpha
            alpha = alpha / 2;
            continue;
        end
        
        % If both conditions are satisfied, break out of the loop
        break;
    end

    % Update w
    w_new = w + alpha * p;
end