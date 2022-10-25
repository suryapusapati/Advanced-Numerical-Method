function [x,f] = multiple2(func,dfunc,ddfunc)
xguess = input('enter initial guess: xguess =');
es = input ('allowable tolerance es =');
maxit = input('maximum number of iterations: maxit =');

iter =1;
x(1) = xguess;
f(1)=feval(func, x(1));
dfdx(1) = feval(dfunc, x(1));
d2fdx2(1) = feval(ddfunc, x(1));
for i=2: maxit
    x(i) = x(i-1) - f(i-1)*dfdx(i-1)/(dfdx(i-1)^2-f(i-1)*d2fdx2(i-1));
    f(i)=feval(func, x(i));
    dfdx(i)=feval(dfunc, x(i));
    d2fdx2(i) = feval(ddfunc, x(i));
    if abs(x(i)-x(i-1))<es
        disp('Newton method has converged'); break;
    end
    iter=i;
end
if (iter >= maxit)
    disp('zero not found to desired tolerance');
end
n = length(x);k=1:n;
disp('    step             x            f            df/dx         d2f/dx2')
out =[k; x; f; dfdx; d2fdx2];
fprintf('%5.0f   %17.14f   %20.15f   %20.15f     %20.15f\n', out)