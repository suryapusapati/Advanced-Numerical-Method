function f = example1_f(x, y)
% Heatequation: d^2T/dx^2=htc*(T-Ta)
% y1' = f1 = y2; y2' = f2 = -htc*(Ta-y1)
htc=0.01; Ta=20;
f1 = y(2); f2 = -htc*(Ta-y(1));
f = [f1 f2]';