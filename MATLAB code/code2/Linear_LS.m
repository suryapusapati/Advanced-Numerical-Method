function s = Linear_LS (x,y)
% linear regression function
% input x and y as row or column vectors
% (they are converted to column form if necessary)

m=length(x); x=x(:); y=y(:);
sx=sum (x); sy=sum(y);
sxx=sum(x.*x); sxy= sum(x.*y);
a0=(sxx*sy-sxy*sx)/(m*sxx-sx^2);
a1=(m*sxy-sx*sy)/(m*sxx-sx^2);
table = [  x   y    (a0+a1*x)     (y-a0-a1*x)];
disp('       x           y      (a0+a1*x)     (y-a0-a1*x)')
disp(table), err = sum(table(:,4).^2)
s(1)=a0; s(2) =a1;
St=sum((y-sy/m).^2); Sr =err;
Syx = sqrt(Sr/(m-2))  % standard error of the estimate
r = sqrt((St-Sr)/St) %correlation coefficient
% plot the regression curve
x1 = min(x); x2=max(x); xx=x1: (x2-x1)/50:x2;
yy=a0+a1*xx; plot (x,y, 'r*', xx,yy, 'g')