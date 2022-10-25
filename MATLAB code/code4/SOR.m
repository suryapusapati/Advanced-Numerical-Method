function x = SOR(A, b, x0, w, tol, max)
% Solution of the system of linear equations Ax=b
% using SOR iterative algorithm
% Outputs: x --- solution vector (n-by-1)

[n,m] = size(A); x=x0; C=-A;
for i= 1:n
    C(i,i)=0;
end
for i=1:n
C(i,1:n) = C(i, 1:n)/A(i,i);
end
for i = 1:n
    r(i,1) =b(i)/A(i,i);
end
i=1;
disp('    i     x1      x2       x3     ....')
while (i<=max)
xold=x;  % save solution form previous step
for j =1:n
    x(j) = (1-w)*xold(j) + w*(C(j,:)*x+r(j));
end
if norm(xold-x)<=tol
    disp('SOR mehtod converged'); return;
end
disp([i      x'])
i =i+1;
end
disp('SOR method did not converge');