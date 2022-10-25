function x = LU_Solve(L, U, b)

% Function to solve the equation LUx=b
%   L --> Lower triangular matrix
%   U --> Upper triangular matrix
%   b --> Rogjt-hand side vector
%   Doolittle decomposition L(i,i)=1
%   Crout decomposition U(i,i)=1

[n m] = size(L); d = zeros(n, 1); x = zeros(n,1);
% solve L d =b using forward substitution
d(1) = b(1)/L(1,1);
for i = 2: n
    d(i) = (b(i) - L(i, 1:i-1)*d(1:i-1))/L(i,i)
end
% Solve U x = d using back substitution
x(n) = d(n)/U(n,n);
for i= n-1:-1:1
    x(i) = (d(i)-U(i,i+1:n)*x(i+1:n))/U(i,i)
end