function [t, y] = Euler_sys(f, tspan, y0,h)
% solve a system of ODE-IVP ousing Euler method
% y' = f(t,y) a <= t <=b
% y = [y1, y2, ..., yn]
% intitial interest is given as tspan = [a, b]
% function f(t,y) returns a column vector of values

a = tspan(1); b = tspan(2); n = (b-a)/h;
t = (a+h:h:b);
k = feval (f,a,y0)';
y(1,:) = y0 + h*k;
for i = 1: n-1
    k=feval(f,t(i),y(i,:))';
      y(i+1,:) = y(i,:)+ h*k;
end
t = [a t]; y =[y0;y]; out = [t' y];
disp('     t           y1        y2       y3  ...')

fprintf('%8.3f  %15.10f  %15.10f\n', out')