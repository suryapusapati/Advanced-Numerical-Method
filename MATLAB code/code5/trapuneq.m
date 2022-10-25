function integr = trapuneq(x,y)
% trapuneq(x,y):
%   Aookues the trapezoidal rule to determine the integral for 
%   n data points (x, y) where x must be in ascending order
% input:
%   x = independent variable
%   y = dependent variable
% output:
%   integr = integral

n = length(x);
if length(y)~=n, error('x and y must be same length'); end
s = 0;
for i = 1: n-1
    if x(i+1) < x(i)
        error('x values must be in ascending order');
    end
    s = s + (x(i+1)-x(i))*(y(i)+y(i+1))/2;
end
integr=s;