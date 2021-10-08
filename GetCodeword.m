function [codewords] = GetCodeword(encoder, info_loc, parity_loc, m)
%ENCODE using c = inv(A)Bm to get parity checks
%   Detailed explanation goes here
codewords = zeros(size(m,1), 200);
[temp, order] = sort([info_loc, parity_loc]);
for i = 1:size(m,1)
    inter = step(encoder, m(i, :)');
    codeword = inter(order);
    codewords(i, :) = codeword;
end
codewords = codewords';
end

