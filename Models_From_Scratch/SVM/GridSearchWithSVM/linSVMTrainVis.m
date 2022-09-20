
function trainAndVisualize(X, y, C)
% this method is created mainly to visualize 
% the effect of C values on the model

model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);
    
end

