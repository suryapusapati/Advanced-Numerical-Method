function t = shoot_secant(fun,tspan,ya,g,t1,t2,max_it,tol)
% Nonlinear shooting method based on secant method
% convert BVP y''=f(x,y,y'); y(a) = ya; g(y(b),y'(b)) = 0
% into IVP    u''=f(x,u,u'); u(a) = ya; u'(a) = t
% update t by secant rule to find zero of the error function:
% m(t) = g(u(b),u'(b)) - residual at x = b
% stop when abs(m(t)) < tol or after max_it iterations
clear all
% ***************************** Define problem
% a = 0; b = 10; max_it = 10; tol = 0.00001;
% ya = 40, yb = 200, g = "z1 - 200';
fun = input ('name of the function = ');
tspan = input('tspan [a b] = ');
ya = input('boundary condition at x=a: ya = ');
g = input('boundary condition at x=b: g(z1, z2) = ');
t1 = input('first initial guess at x=a: t1 = ');
t2 = input('second initial guess at x=a: t2 = ');
max_it = input('maximum number of iterations max_it = ');
tol = input('tolerance tol = ');
% ***********************************
t(1) = t1; t(2) = t2;   % start with initial guesses t1 and t2
test = 1; i = 1; hold on
while (test > tol) && (i <= max_it)
    if i > 2
        t(i) = t(i-1) - (t(i-1)-t(i-2))*m(i-1)/(m(i-1) -m(i-2));
    end
    z0 = [ya t(i)]; [x,z] = ode23(fun, tspan, z0);
    [n nn] = size(z); z1 = z(n,1); z2 = z(n, 2);
    m(i) = z1-g; H = plot(x,z); test = abs(m(i)); i = i+1;
end
t
[x z]
hold off
