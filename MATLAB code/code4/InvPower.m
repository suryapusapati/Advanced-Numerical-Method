function [z,m] = InvPower(A, max_it, tol)
[n,nn] = size(A); z=ones(n,1);
it=0; error=100;[L,U]=LU_factor(A);

while (it < max_it && error > tol)
    w =LU_Solve(L,U,z); ww = abs(w);
    [k,kk] = max(ww); % kk is index of max wlement of ww
    m = (z'*z)/(z'*w); % estimate of eigenvalues
    z=w/w(kk);  % estimate of eigenvector
    out = [ it+1     m     z']; disp(out)
    error = norm(A*z-m*z);
    it=it+1;
end
error