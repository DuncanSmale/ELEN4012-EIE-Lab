function H = GenerateLDPC(n,j,i)
%This code is to generate parity check matrix of LDPC code using Gallager's construction.
%Date: November 23, 2013.
% n: Number of columns
% j: Number of 1's per column
% i: Number of 1's per row
w_c = j; % Column weight
w_r = i; % Row weight
k = n*w_c/w_r; % Number of rows
H_sub = zeros(n/w_r,n); % First sub-matrix; there are w_c such sub-matrices.
%% Generation of Basic Sub-matrix
for i = 1:n/w_r
    for j = (i-1)*w_r+1:i*w_r
        H_sub(i,j) = H_sub(i,j) + 1;
    end
end
%% Permutation of Basic Sub-matrix
H_pre = H_sub;
for t = 2:w_c
    x = randperm(n);
    H_sub_perm = H_sub(:,x);
    H_pre = [H_pre H_sub_perm];
end
H = zeros(k,n);
for p = 1:w_c
    H((p-1)*(n/w_r)+1:(p)*(n/w_r),1:n) = H((p-1)*(n/w_r)+1:(p)*(n/w_r),1:n) + H_pre(:,(p-1)*n+1:p*n);
end
% H is the requires parity chack matrix.
end

