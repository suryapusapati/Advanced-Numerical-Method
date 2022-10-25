function [x,f] = multiple1(func,dfunc)
m = input('enter multiplicity of the root =');
xguess = input('enter initial guess: xguess =');
es = input ('allowable tolerance es =');
maxit = input('maximum number of iterations: maxit =');

iter =1;
x(1) = xguess;
f(1)=feval(func, x(1));
dfdx(1) = feval(dfunc, x(1));
for i=2: maxit
    x(i) = x(i-1) - m*f(i-1)/dfdx(i-1);
    f(i)=feval(func, x(i));
    dfdx(i)=feval(dfunc, x(i));
    if abs(x(i)-x(i-1))<es
        disp('Newton method has converged'); break;
    end
    iter=i;
end
if (iter >= maxit)
    disp('zero not found to desired tolerance');
end
n = length(x);k=1:n;
disp('    step      x       f      df/dx')
out =[k; x; f; dfdx];
fprintf('%5.0f   %20.14f   %21.15f   %21.15f\n', out)