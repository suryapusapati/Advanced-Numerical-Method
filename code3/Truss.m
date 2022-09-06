function [A, b]=Truss(alpha, beta, gamma, delta)

A = zeros(10,10);
A(1,1) = 1; A(1,5) = sin(alpha);
A(2,2) = 1; A(2,4) = 1; A(2,5) = cos(alpha);
A(3,7) = sin(beta); A(3,8) = sin(gamma);
A(4,4) = -1; A(4,6) = 1; A(4,7) = -cos(beta); 
A(4,8) = cos(gamma);
A(5,3)= 1; A(5,9) = sin(gamma);
A(6,6) = -1; A(6,9) = -cos(delta);
A(7,5) = -sin(alpha); A(7,7)=-sin(beta);
A(8,5) = -cos(alpha); A(8,7) = cos(beta); A(8,10)=1;
A(9,8) = -sin(gamma); A(9,9) = -sin(delta);
A(10,8) = -cos(gamma); A(10,9) = cos(delta); A(10,10) = -1;

b = zeros(10,1); b(3,1)=100;
f = A\b
