function a = Gauss_Newton (x,y)

% Nonlinear Regression of f(x)=exp(-a0*x)cos(a1*x)
% using Gauss-Newton method

a = input('Enter the initial guesses [a0, a1] =');
tol = input('Enter the tolerance to1 = ');
itmax = input ('Enter the maximum iteration number itmax = ');

n=length(x)
disp('     iter      a0      a1      da0       da1')
for iter=1:itmax
    a0=a(1); a1=a(2);
    for i=1:n
        f(i)=exp(-a0*x(i)).*cos(a1*x(i));
        d(i)=y(i)-f(i);
        z(i,1)=-x(i).*exp(-a0*x(i)).*cos(a1*x(i));
        z(i,2)=-x(i).*exp(-a0*x(i)).*sin(a1*x(i));
    end
    da=(z'*z)\(z'*d'); a=a+da';
    out=[iter   a   da']; disp(out)
    if (abs(da(1)) < tol & abs(da(2)) < tol)
        disp('Gauss-Newton method has converged'); break
    end
end

x1=min(x); x2=max(x); xx=x1:(x2-x1)/50:x2;
yy=exp(-a0*xx).*cos(a1*xx);
H=plot(xx,yy,x,y,'ro');
set(H,'LineWidth',2,'MarkerSize',7);

      