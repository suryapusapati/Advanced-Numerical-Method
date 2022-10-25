function [t, y] = Midpoint(f, tspan, y0,h)
% function [t,y] = midpoint(f, tspan, y0,h)
%solve y' = f(t,y)
%with intitial condition y(a)=y0
% using n steps of the midpoint (RK2) method;

a = tspan(1); b = tspan(2); n= (b-a)/h;
t= (a+h:h:b);
k1 = feval (f,a,y0);
k2 = feval(f, a+h/2, y0+k1/2*h);
y(1) = y0 + k2*h;
for i = 1: n-1
    k1=feval(f,t(i),y(i));
    k2=feval(f,t(i)+h/2,y(i)+k1/2*h);
    y(i+1) = y(i)+ k2*h;
end
t = [a t]; y =[y0 y];
disp('    step        t           y')
k = 1: length(t); out = [k; t; y];
fprintf('%5d  %15.10f  %15.10f\n', out)