function x = fixed_pt_sys(G, x0, es, maxit)
% Solve the nonlinear system x = G(x) using fixed-point method
% Vectors x and x0 are row vectors (for display purposes)
% function G returns a column vector, [g1(x), ..gn(x)]'
% stop if norm of change in solution vector is less than es
% y = feval(G,xold); the next approximate solution is xnew = y';

disp([0    x0 ]);      %display initial estimate
xold = x0;
iter = 1;
while (iter <= maxit)
     y = feval(G, xold);
     xnew = y';
     dif = norm(xnew - xold);
     disp([iter    xnew    dif]);
     if dif <= es
        x = xnew;
        disp('Fixed-point iteration has converged')
        return;
     else
        xold = xnew;
     end
     iter = iter + 1;
end
disp('Fixed-point iteration did not converge')
x=xnew;
