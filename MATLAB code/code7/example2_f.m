function f = example2_f(t,y)
% Heatequation: d^2T/dx^2=htc*(T^4-Ta^4)
% y1' = f1 = y2; y2' = f2 = -htc*(Ta^4-y1^4)
htc=5*10^(-8); Ta=20;
f1 = y(2); f2 = -htc*(Ta^4-y(1)^4);
f = [f1 f2]';