function X_rec = recoverData(Z, U, K)
%recoverData returns an approximation of the original dataset
% given the reduced dataset, the U matrix and the number of priciple
% components.

X_rec = zeros(size(Z, 1), size(U, 1));
U_red = U(:, 1:K);
X_rec = Z * U_red';

end
