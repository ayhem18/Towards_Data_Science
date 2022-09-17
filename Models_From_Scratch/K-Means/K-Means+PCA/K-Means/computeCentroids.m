function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the newly computed centroids. Each new centroid
% corresponds to the mean of the points at each group

% X: with m rows and n columns, where each row represents an example: a point

% idx: object where the i-th value represents the index of the centroid
% closest to the i-th example in X.

% K: number of centroids

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


X_inv = X';
index_col = idx(:); % have the indices stored as a column vector
for i=1:K
  %find the number of examples grouped with the current centroid
  num_i = sum(idx == i);
  % multiply the inverse of X by a boolean array.
  % the i-th row of the resulting matrix will represent
  % the sum of elements for which the corresponding index is equal to 'i']
  % dividing by the number of such elements: num_i to obtain the mean.
  centroids(i,:) = X_inv * (index_col == i) / num_i;
endfor  

end

