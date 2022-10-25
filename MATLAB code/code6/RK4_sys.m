function [t, y] = RK4_sys(f, tspan, y0,h)
% solve a system of ODEs using 4th-order RK method
% input: column vector t and row vector y
% return: column vector of values for y'

a = tspan(1); b = tspan(2); n = (b-a)/h;
t = (a+h:h:b)';
k1 = feval (f,a,y0)';
k2 = feval(f, a+h/2, y0+k1/2*h)';
k3 = feval(f, a+h/2, y0+k2/2*h)';
k4 = feval(f, a+h, y0+k3*h)';
y(1,:) = y0 + (k1/6+k2/3+k3/3+k4/6)*h;
for i = 1: n-1
    k1=feval(f,t(i),y(i,:))';
    k2=feval(f,t(i)+h/2,y(i,:)+k1/2*h)';
    k3=feval(f,t(i)+h/2,y(i,:)+k2/2*h)';
    k4=feval(f,t(i)+h,y(i,:)+k3*h)';
    y(i+1,:) = y(i,:)+ (k1/6+k2/3+k3/3+k4/6)*h;
end
t = [a; t]; y =[y0;y]; out = [t y];
disp('     t           y1        y2       y3  ...')

fprintf('%8.3f  %15.10f  %15.10f\n', out')