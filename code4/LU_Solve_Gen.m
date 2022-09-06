function x = LU_Solve_Gen(L, U, B)
% Function to solve the equation LUx=B
% L --> Lower triangualr matrix (1's on diagonal)
% U --> Upper triangualr matrix (1's on diagonal)
% B --> Right-hand-side matrix

[n n2] = size(L); [m1 m] = size(B);
% Solve Ld = B using forword substritution
for j =1:m
    d(1,j) =B(1,j);
    for i = 2:n
        d(i,j) = B(i,j) - L(i, 1:i-1)*d(1:i-1,j);
    end
end

%sove Ux=d using back substitution
for j =1:m
    x(n,j) = d(n,j) / U(n,n);
    for i = n-1:-1:1
        x(i,j)=(d(i,j)-U(i,i+1:n) * x(i+1:n,j))/U(i,i);
    end
end

