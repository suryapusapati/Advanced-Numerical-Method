function z = Cubic_LS(x,y)

% Cubic Least Squares regression function
% input x and y as row or column vectors
% (they are converted to column form if necessary)

n = length(x); x=x(:); y=y(:);
sx=sum(x); sx2=sum(x.^2); sx3=sum(x.^3); 
sx4=sum(x.^4);sx5=sum(x.^5); sx6=sum(x.^6);
sy=sum(y); syx=sum(x.*y); 
syx2=sum(y.*x.^2);syx3=sum(y.*x.^3);
A=[n   sx  sx2  sx3
   sx  sx2 sx3  sx4
   sx2 sx3 sx4  sx5
   sx3 sx4 sx5  sx6];
r=[sy      syx      syx2    syx3]';
z=(A\r)'; a0=z(1); a1=z(2);a2=z(3);a3=z(4);
p = a0 +a1*x + a2*x.^2 + a3*x.^3;
table = [x   y    p    (y-p)];
disp('      x     y       p(x) = a0 +a1*x + a2*x.^2 + a3*x.^3  y-p(x)')
disp (table), err = sum(table(:,4).^2)

St = sum((y-sy/n).^2); Sr=err;
Syx=sqrt(Sr/(n-3)) % standard error of the estimate
r = sqrt((St-Sr)/St) % correlation coefficient

%plot the regression curve

x1=min(x); x2=max(x); xx= x1:(x2-x1)/50:x2;
yy=a0+a1*xx+a2*xx.^2+a3*xx.^3;
plot (x,y,'r*',xx,yy,'b');