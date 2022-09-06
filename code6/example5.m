function f = example5(t,y)
% dy1/dt = f1 = -0.5 y1
% dy2/dt = f2 = 4 - 0.1*y1 - 0.3*y2
% let y(1) = y1, y(2) = y2
% tspan = [0 1]
% initial conditions y0 = [4, 6]
f1 = -0.5*y(1);
f2 = 4 - 0.1*y(1) - 0.3*y(2);
f = [f1, f2]';
