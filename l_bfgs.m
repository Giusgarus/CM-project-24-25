function [w] = l_bfgs(X, w, y, m, epsilon, limit)
    % L-BFGS optimization function
    %
    % Inputs:
    %   - X: Matrix of input data
    %   - w: Initial weight (parameter) vector
    %   - y: Vector of target values
    %   - m: Memory size for L-BFGS (number of corrections stored)
    %   - epsilon: Stopping criterion threshold based on gradient norm
    %   - limit: Maximum number of iterations to store in S and Y
    %
    % Output:
    %   - w: Updated weight vector after the L-BFGS optimization

    % Initialize iteration counter
    k = 1;
    
    % Allocate space for S and Y:
    %   - S: Stores changes in w (s_k = w_{k+1} - w_k)
    %   - Y: Stores changes in gradient (y_k = g_{k+1} - g_k)
    S = zeros(m, limit);
    Y = zeros(m, limit);

    % Compute initial gradient and set up search direction
    p = - compute_gradient(X, w, y);  % p is the search direction
    g = -p;                           % g is the current gradient

    % Choose an initial step size alpha (could be determined by line search)
    alpha = 0.001;
    
    % Compute an initial update to w using the chosen alpha
    w1 = w + alpha * p;
    
    % Recompute the gradient for the new w
    p = -compute_gradient(X, w1, y);
    
    % Store the first s and y values
    S(1:end, 1) = w1 - w;     % s_1
    Y(1:end, 1) = -p - g;     % y_1
    k = k + 1;
    
    % Update the stored gradient
    g = -p;

    % Iterate until the gradient norm is below epsilon
    while norm(g) > epsilon
        
        % Save current gradient in a temporary variable
        temp = g;

        % Compute the new search direction p using two_loop
        p = -two_loop(S, Y, g, m, k);

        % Find step size alpha via line search
        % alpha = line_search(p,w,temp);
        % alpha = backtracking_search(g,X,w,p,y);
        alpha = lineSearchStrongWolfe(X, y, w, p, g, 0.0001, 0.9);

        % Update w based on the search direction and step size
        s = alpha * p;
        w = w + s;

        % Recompute the gradient with the updated w
        g = compute_gradient(X, w, y);

        % Update S and Y. If we've filled them up, discard the oldest entries.
        if k <= limit
            S(1:end, k) = s;
            Y(1:end, k) = g - temp;
        else
            % Shift existing columns to the left
            S(1:end, 1:limit-1) = S(1:end, 2:end);
            S(1:end, limit)     = s;
            Y(1:end, 1:limit-1) = Y(1:end, 2:end);
            Y(1:end, limit)     = g - temp; 
        end

        % Increment iteration counter
        k = k + 1;
    end
end