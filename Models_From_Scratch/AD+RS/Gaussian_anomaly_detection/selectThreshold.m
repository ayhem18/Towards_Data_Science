function [bestEpsilon bestF1] = selectThreshold(yval, pval)
% Given a cross validation data, this function returns the best f1 score
%% achieved as well as the corresponding epsilon.
%% The F1 score is chosen as the performance metric as it
%% reflects the model's capacity to detect both anamalies and
%% non-anomalies


best_f1 = 0;
best_eps = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    predictions = (pval <= epsilon); % set the anomalies as one
     
    false = sum(predictions != yval);

    tp = sum((yval == predictions) .* (yval == 1));
    
    f1 = 2 / (2 + false / tp);
    
   if f1 > best_f1
     best_f1 = f1;
     best_eps = epsilon;
   endif

endfor

bestF1 = best_f1;
bestEpsilon = best_eps;
end
