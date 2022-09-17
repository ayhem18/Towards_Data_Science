function Z = projectData(X, U, K)
% PROJECTDATA computes the reduced representation of X using the top 
% K principle components: the first K columns of U

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

U_red = U(:, 1:K);
Z = X * U_red ;

end
