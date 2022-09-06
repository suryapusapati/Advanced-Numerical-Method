function [L,U]=LU_factor(A,unitdiag);
%%  function [L,U]=lufactor(A,unitdiag);
%%  
%%  DIRECT LU FACTORIZATION OF THE MATRIX A,     A=LU where L is lower
%%     triangular, U is upper triangular and one of L,U has unit diagonal
%%
%%  INPUT:
%%    A   - square matrix to be factored 
%%    unitdiag -  input  'L' (' ' required) if L is to have unit diagonal
%%             -  input  'U' (' ' required) if U is to have unit diagonal
%%             - if only A is input, then L has unit diagonal by default  
%%  OUTPUT:
%%    L  -  lower triangular matrix
%%    U  - upper triangular matrix

if nargin==1;       %% determine which matrix
   lflag=1;                            %% is to have unit diagonal
elseif    unitdiag=='L'
   lflag=1;
else
   lflag=0;
end;

n=length(A(:,1));  % determine the size of the system
flag=0;            % initialize stopping flag
L=zeros(n,n);U=zeros(n,n);  % Start with zero matrices to be filled in 

%% Step 1
if A(1,1)==0;
   flag==1;
end;

if flag==0;
    if lflag==1;
        L(1,1)=1;U(1,1)=A(1,1);
    else
        U(1,1)=1;L(1,1)=A(1,1);
    end;
    for j=2:n; %step 2
      U(1,j)=A(1,j)/L(1,1);
      L(j,1)=A(j,1)/U(1,1);
    end;
end;
if flag==0;
   for i=2:n-1;  % Step 3
       if lflag==1;
           L(i,i)=1; U(i,i)=A(i,i)-L(i,1:i-1)*U(1:i-1,i);
       else
           U(i,i)=1; L(i,i)=A(i,i)-L(i,1:i-1)*U(1:i-1,i);
       end;
       if (U(i,i)==0 | L(i,i)==0);   % Stop if a diagonal of L or U is zero
              flag=1;
       end;
       if flag==0;
          for j=i+1:n;
            U(i,j)=(A(i,j)-L(i,1:i-1)*U(1:i-1,j))/L(i,i);
            L(j,i)=(A(j,i)-L(j,1:i-1)*U(1:i-1,i))/U(i,i);
          end;
       end;
    end;
    if flag==0;
      if lflag==1;
          L(n,n)=1;U(n,n)=A(n,n)-L(n,1:n-1)*U(1:n-1,n);
      else
          U(n,n)=1;L(n,n)=A(n,n)-L(n,1:n-1)*U(1:n-1,n);
      end;
    end;
end;
if flag==1,
   L='Factorization impossible'
   U=' ';  
end;