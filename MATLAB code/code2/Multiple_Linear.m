function z = Multiple_Linear(x1,x2,y)

% Multiple variable Least Squares regression function
% input x and y as row or column vectors

n = length(x1); x1=x1(:); x2=x2(:); y=y(:);
sx1=sum(x1); sx2=sum(x2); sx1x2=sum(x1.*x2);
sx1x1=sum(x1.^2);sx2x2=sum(x2.^2); 
sy=sum(y); sx1y = sum(x1.*y); sx2y=sum(x2.*y);
A=[n   sx1  sx2; sx1 sx1x1 sx1x2; sx2 sx1x2 sx2x2]; 
r=[sy      sx1y      sx2y]';
z=(A\r)'; a0=z(1); a1=z(2);a2=z(3);

table = [x1   x2    y    (a0+a1*x1+a2*x2)  (y-a0-a1*x1-a2*x2)];
disp('    x1   x2    y    (a0+a1*x1+a2*x2)  (y-a0-a1*x1-a2*x2)')
disp (table), err = sum(table(:,5).^2)

St = sum((y-sy/n).^2); Sr=err;
Syx=sqrt(Sr/(n-3)) % standard error of the estimate
r = sqrt((St-Sr)/St) % correlation coefficient

%plot the experimental data and regression plane
H=plot3(x1,x2,y,'ro'); grid on; set(H,'LineWidth',6);
H1=xlabel('cure time (days)'); set (H1,'FontSize',12);
H2=ylabel('Water Content'); set (H2,'FontSize',12);
H3=zlabel('Strength (psi)'); set (H3,'FontSize',12); hold on;
x1a=min(x1); x1b=max(x1); x1s=x1a:(x1b-x1a)/50:x1b;
x2a=min(x2); x2b=max(x2); x2s=x2a:(x2b-x2a)/50:x2b;
[xx1,xx2]=meshgrid(x1s,x2s);
yy=a0+a1*xx1+a2*xx2; surf(xx1,xx2,yy); hold off