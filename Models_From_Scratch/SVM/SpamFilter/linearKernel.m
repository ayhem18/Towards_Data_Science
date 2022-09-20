function sim = linearKernel(x1, x2)
%This function returns a linear kernet between two vectors
% convert the vectors into column vectors (if they are not)
x1 = x1(:); x2 = x2(:);

% Compute the kernel
sim = x1' * x2; 

end