function RBFSVMTrainVis(X, y, C, sigma)
% this method is created mainly to visualize 
% the effect of C values on the RBF SVM model

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

end
