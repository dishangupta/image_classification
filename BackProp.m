% The Backpropagation algorithm for NN training
% Input: Labels (Y) = {1,-1}, Data(X)
% Output: Weights (W) after training

function [Wji, Wkj] = BackProp (Y, X, Ytest, Xtest)

% Data parameters
D = size(X, 1); % # data points
X = [ones(D, 1) X]; % append 1 for bias
N = size(X, 2); % # features

% Construct NN
I = size(X, 2) - 1; % # neurons in input layer
J = 600; % # neurons in hidden layer
K = max(Y) + 1; % # neurons in output layer

% Initialize weights randomly between 0 and 1
Wji = (0.01)*(-1 + 2*rand(J, I + 1)); % +1 for bias
Wkj = (0.01)*(-1 + 2*rand(K, J + 1)); % +1 for bias

% Store weights for previous iteration
Wji_old = -Inf*ones(J, I + 1);
Wkj_old = -Inf*ones(K, J + 1);

% Kernel Parameters -- Gaussian
Var = 1;

% Learning parameters
eta = 0.1 ; % learning rate
eps = 0.001; % convergence criteria
maxIter = 1000;

tic
for iter = 1:maxIter
    for d = 1:D
        
        % Peform Feed-Forward Pass
        Zj = Wji*X(d, :)';
        Aj = 1./(1 + exp(-Zj));
        Aj = [1; Aj]; % append 1 for bias
        Zk = Wkj*Aj;
        Yk = 1./(1 + exp(-Zk));
        
        % Perform error backpropagation
        T = zeros(K, 1); % true output desired at output layer
        T(Y(d) + 1) = 1; % detect digit by index, set to 1
        
        delK = Yk - T;
        delJ = Aj.*(1-Aj).*((Wkj)'*delK);
        
        % Update weights hidden layer
        %%%% SOM neighborhood update
        %%%% Hidden layer, 1-D case
        for k = 1:K
            for j = 1:J+1
                Wkj(k, j) = Wkj(k, j) - eta*(delK(k)*Aj(j));
                if (j > 1)
                    Wkj(k, j-1) = Wkj(k, j-1) - eta*(0.0540/0.3989)*delK(k)*Aj(j);
                end
                if (j > 2)
                    Wkj(k, j-2) = Wkj(k, j-2) - eta*(0.0044/0.3989)*delK(k)*Aj(j);
                end
                if (j < J+1)
                    Wkj(k, j+1) = Wkj(k, j+1) - eta*(0.0540/0.3989)*delK(k)*Aj(j);
                end
                if (j < J)
                    Wkj(k, j+2) = Wkj(k, j+2) - eta*(0.0044/0.3989)*delK(k)*Aj(j);
                end
            end
        end
        %Wkj = Wkj - eta*(delK*Aj');
        %Lkj = circshift(delK*Aj', [0 -1]); % Update from lower neurons
        %Lkj(:, end) = 0;
        %Ukj = circshift(delK*Aj', [0 1]); % Update from upper neurons
        %Ukj(:, 1) = 0;
        %Wkj = Wkj - 0.5*eta*(Lkj + Ukj);
        
        % Update weights input layer
        Wji = Wji - eta*(delJ(2:end)*X(d, :));
        
    end
    
    sum(sum(abs(Wji - Wji_old))) + sum(sum(abs(Wkj - Wkj_old)))
    % Check convergence
    if (sum(sum(abs(Wji - Wji_old))) + sum(sum(abs(Wkj - Wkj_old))) < eps)
        break;
    end
    
    iter
    % Update previous iteration weights
    Wji_old = Wji;
    Wkj_old = Wkj;
    
    pct_err = ClassificationError(Ytest, Xtest, Wji, Wkj)
end

toc
%{
figure;
h = plot(1:100, err_iter(:, 1), 1:100, err_iter(:, 2), 1:100, err_iter(:, 3));
set(h, 'LineWidth', 2);
hold on
legend('J = 2', 'J = 32', 'J = 200');
title('DataSet = Training Set 4, Eta = 0.1');
xlabel('# Iterations');
ylabel('% Error');
%}

%iter

pct_err = ClassificationError(Ytest, Xtest, Wji, Wkj)

end

