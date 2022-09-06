function [t,y] = Heun_iter(f, tspan, y0, h, itmax)
% solve y' = f(t,y) with intitial condition y(a) = y0
% using n steps of the Heun's method;
% itmax = 0: Heun's mehtod without iterative correction

a = tspan(1); b= tspan(2); n = (b-a)/h;
t = (a+h:h:b);
k1 = feval(f,a,y0);
k2 = feval(f, a+h, y0+k1*h);
y(1) = y0 + 0.5*(k1+k2)*h;
for iter =1: itmax
    k2 = feval (f, a+h, y(1));
    y(1)= y0 + 0.5*(k1+k2)*h
end
for i=1: n-1
    k1 = feval(f,t(i), y(i));
    k2 = feval(f, t(i)+h, y(i)+k1*h);
    y(i+1) = y(i) + 0.5*(k1+k2)*h;
    for iter =1: itmax
        k2 = feval(f, t(i)+h, y(i+1));
        y(i+1)=y(i)+0.5*(k1+k2)*h
    end
end
t = [a  t];
y = [y0 y];
