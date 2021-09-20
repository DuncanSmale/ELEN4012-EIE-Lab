function [x] = GetCodeword(A_inv,B_gf, m)
%ENCODE using c = inv(A)Bm to get parity checks
%   Detailed explanation goes here
c = A_inv * B_gf * m';
x_gf = [m'; c];
x = double(x_gf.x);
end

