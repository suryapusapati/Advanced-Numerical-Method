function [x,y] = linear_FD
% script for finite-difference ODE-BVP
% y_xx = p(x) y_x +q(x) y +r(x)    aa<=x<=bb
% y(aa) = ya; y(bb) = yb
% ************************* begin problem definition
clear all
aa = input('left boundary aa = ');
bb = input('right boundary bb = ');
n = input('number of subintervals n = ');
% boundary conditions
ya = input('left boundary condition ya = ');
yb = input('right boundary condition yb = ');
% define p(x), q(x), r(x)
% Note: use p(x) ='c*x.^0' if p(x) = constant,
%           p(x) ='0*x' if p(x) =0
px= input('function p(x) = ');
qx= input('function q(x) = ');
rx= input('function r(x) = ');

h = (bb-aa)/n; h2 = h/2; hh = h*h; % define parameters
x = linspace(aa+h, bb, n);          % grid pionts x(1),......, x(n-1)
pp = eval(px); qq = eval(qx); rr =eval(rx);
p = pp(1:n-1); q =qq(1:n-1); r = rr(1:n-1);
% upper diagonal (a), diagonal (d), lower diagonal (b)
a = zeros(1, n-1); b =a;
a(2:n-1) = 1 - p(1,2: n-1)*h2; d = -(2+hh*q);
b(1:n-2) = 1 + p(1,1: n-2)*h2;
c(1) = hh*r(1) - (1+p(1)*h2)*ya;   % right-hand side (c)
c(2:n-2) = hh*r(2:n-2);
c(n-1) = hh*r(n-1)-(1-p(n-1)*h2)*yb;
y = Tridiag (a, d, b, c)
xx = [aa x]; yy = [ya y yb];
out = [xx' yy']; disp(out)
plot(xx,yy,'b-o');
