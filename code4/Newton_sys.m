function x = Newton_sys(F, JF, x0, tol, maxit)

% Solve the nonlinear system F(x) = 0 using Newton's method
% Vectors x and x0 are row vectors (for display purposes)
% function F returns a column vector, [f1(x), ..fn(x)]'
% stop if norm of change in solution vector is less than tol
% solve JF(x) y = - F(x) using Matlab's "backslash operator"
% y = -feval(JF, xold) \ feval(F, xold);
% the next approximate solution is x_new = xold + y';

xold = x0;
disp([0 xold ]);
iter = 1;
while (iter <= maxit)
     y= - feval(JF, xold) \ feval(F, xold);
     xnew = xold + y';
     dif = norm(xnew - xold);
     disp([iter    xnew    dif]);
     if dif <= tol
        x = xnew;
        disp('Newton method has converged')
        return;
     else
        xold = xnew;
     end
     iter = iter + 1;
end
disp('Newton method did not converge')
x=xnew;