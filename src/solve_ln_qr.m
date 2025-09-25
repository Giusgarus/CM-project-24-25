function [b] = solve_ln_qr(b, V)
    % Obtain the dimensions of V: the tilde (~) ignores the number of rows,
    % while 'n' is the number of columns
    [~, n] = size(V);

    % Loop over each column of V
    for i = 1:n
        % Extract the i-th column of V
        v = V(:, i);

        % Update the vector b:
        %   - v' * b computes the dot product between v and b
        %   - 2 * v * (v' * b) computes the Householder reflection term
        %   - subtract that term from b
        b = b - 2 * v * (v' * b);
    end
end