function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS, given a set of points (examples) as well as current
% centroids finds the closest centroid to each example.
% the function returns a list of indices: where the i-th index corresponds to
% index of the centroid closest to the i-th example

% Set K: number of centroids
K = size(centroids, 1);

% the list of indices
idx = zeros(size(X,1), 1);

m = size(X, 1);

% loop through the set of examples
for x=1:m
  % calculate the distance to the first centroid
  error = X(x, :) - centroids(1, :);
  % save the current minimum distance to min_dist
  min_dist = error * error';
  best_c = 1; % best_c stores the index of the closest centroid
  for c=2:K
    % calculate the distance to the c-th centroid
    error = X(x, :) - centroids(c, :);
    dist =  error * error';
    % if the new distance is less than the current minimal distance, then
    % update the corresponding variables
    if dist <= min_dist
      min_dist = dist;
      best_c = c;
    endif
  endfor
  idx(x) = best_c;
endfor
end

