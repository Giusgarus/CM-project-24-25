function [V, H] = householder(A)
    % HOUSEHOLDER Computes the Householder transformation of a matrix.
    % This function transforms a given matrix A into an upper triangular
    % matrix H using Householder reflectors. It also returns the matrix V
    % containing the reflector vectors.
    %
    % Inputs:
    %   A - The input matrix to be transformed.
    %
    % Outputs:
    %   V - A matrix where each column contains a Householder reflector vector.
    %   H - The upper triangular matrix obtained after applying the reflectors.

    % Determine matrix dimensions
    [m, n] = size(A);
    
    % Allocate space for the reflectors and copy A for transformations
    V = zeros(m, n); % Matrix to store Householder vectors
    H = A;           % Copy of A to apply transformations
    
    % Loop over columns to create Householder reflectors
    for j = 1:n
        % Extract the relevant part of the column
        x = H(j:end, j);
        
        % Compute the reflector’s magnitude (norm of the vector)
        sigma = norm(x);
        
        % Define the reflector vector
        v = x;
        v(1) = v(1) + sign(x(1)) * sigma; % Modify the first element of v
        
        % Normalize the reflector vector
        v = v / norm(v); % Ensure the reflector has unit norm
        
        % Store the reflector in the corresponding column of V
        V(j:end, j) = v;
        
        % Apply the reflector to the remaining columns of H
        % H = H - 2 * v * (v' * H), applied to the submatrix
        H(j:end, j:end) = H(j:end, j:end) - 2 * (v * (v' * H(j:end, j:end)));
    end
end
