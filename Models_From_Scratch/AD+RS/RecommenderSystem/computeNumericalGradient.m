function numgrad = computeNumericalGradient(J, theta)
% this function returns a numerical approximation of gradient of the 
% function J, with respect to the vector Theta. using the finite differences
% method. The difference between the numerical gradient and the analytical 
% one should not exceed 10 ^ -9.

numgrad = zeros(size(theta));
perturb = zeros(size(theta));
e = 1e-4;
for p = 1:numel(theta)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = J(theta - perturb);
    loss2 = J(theta + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
end

end
