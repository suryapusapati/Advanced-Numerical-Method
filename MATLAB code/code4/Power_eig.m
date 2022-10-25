function [z,m] = Power_eig(A, max_it, tol)
[n,nn] = size(A); z=ones(n,1);
it=0; error=100;
disp('      it         m       z(1)      z(2)       z(3)      z(4)      z(5)')
while (it < max_it && error > tol)
    w =A*z; ww = abs(w);
    [k,kk] = max(ww); % kk is index of max wlement of ww
    m = w(kk); % estimate of eigenvalues
    z=w/w(kk);  % estimate of eigenvector
    out = [ it+1     m     z']; disp(out)
    error = norm(A*z-m*z);
    it=it+1;
end
error