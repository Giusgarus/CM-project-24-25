function [r] = two_loop(S, Y, g, dim, k)
    % Implements the two-loop recursion used in L-BFGS to compute a search direction.
    % 
    % Inputs:
    %   - S: Matrix containing past updates in the solution space (columns are s-vectors).
    %   - Y: Matrix containing past updates in the gradient space (columns are y-vectors).
    %   - g: Current gradient.
    %   - dim: Dimension of the parameter space.
    %   - k: Current iteration index.
    %
    % Output:
    %   - r: The computed result of the two-loop recursion, used as an approximate search direction.

    % Get the number of columns in S (and Y)
    [~, m] = size(S);

    % Initialize Rho and alpha as zero-arrays of length 'dim'
    Rho = zeros(1, dim);
    alpha = zeros(1, dim);

    % Initialize q as the current gradient
    q = g;

    % Define t as k. If k > m, set t = m. In both cases, choose the columns (t-1) of S and Y.
    t = k;
    if k > m
        t = m;
        sk = S(:, t-1);
        yk = Y(:, t-1);
    else
        sk = S(:, t-1);
        yk = Y(:, t-1);
    end

    % Compute gamma based on the dot products of sk and yk
    gamma = (sk' * yk) / (yk' * yk);

    % Build the initial Hessian approximation as gamma times the identity
    H0k = gamma * eye(dim);

    % First loop (from t-1 down to 1):
    %   - Compute Rho(i) = 1 / (y' * s)
    %   - Compute alpha(i) = Rho(i) * (s' * q)
    %   - Update q = q - alpha(i)*y
    for i = t-1:-1:1
        y = Y(:, i);
        s = S(:, i);
        Rho(i) = 1 / (y' * s);
        alpha(i) = Rho(i) * (s' * q);
        q = q - alpha(i) * y;
    end

    % Multiply the result of the first loop by the initial Hessian approximation
    r = H0k * q;

    % Second loop (from 1 to t-1):
    %   - Compute beta = Rho(i) * (y' * r)
    %   - Update r = r + s * (alpha(i) - beta)
    for i = 1:t-1
        y = Y(:, i);
        s = S(:, i);
        beta = Rho(i) * (y' * r);
        r = r + s * (alpha(i) - beta);
    end
end