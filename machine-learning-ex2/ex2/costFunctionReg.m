function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sigmoid_var = (X * theta );

grad = ((1/m) * (X' * ((sigmoid(sigmoid_var)) - y))) + ((lambda / m) .* theta);
_grad = (1/m) * (X' * ((sigmoid(sigmoid_var)) - y));
grad(1) = _grad(1);

_J1 = (-1) * y' * log(sigmoid(sigmoid_var));
_J2 = (1 - y)' * (log(1 - sigmoid(sigmoid_var)));

_J = (_J1 - _J2) / m;

theta(1) = 0;
_thetasq = theta' * theta;
_JL = (lambda / (2 * m)) * _thetasq;
J = _J + _JL;

% =============================================================

end
