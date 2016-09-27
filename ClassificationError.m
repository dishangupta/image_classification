% This function computes the classification error
% Input: Y, W, X
% Output: Percentage classfication error

function [pct_err] = ClassificationError(Y, X, Wji, Wkj)

D = size(X, 1); % # data points

% Check if 1 is appended
if (size(Wji, 2) > size(X, 2))
    X = [ones(D, 1) X];
end

pct_err = 0;
for d = 1:D

% Peform Feed-Forward Pass
    Zj = Wji*X(d, :)';
    Aj = 1./(1 + exp(-Zj));
    Aj = [1; Aj]; % append 1 for bias
    Zk = Wkj*Aj;
    Yk = 1./(1 + exp(-Zk));
    
    o = 0;
    [maxVal, maxInd] = max(Yk); % extract index at maximum probability
    o = maxInd - 1;
    
    pct_err = pct_err + (Y(d) ~= o); 
    
    
end

pct_err = 100*pct_err/D


end

