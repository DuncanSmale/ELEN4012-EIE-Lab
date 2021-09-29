function isCodeword = CheckCodeword(H_gf,x_gf)
%CHECKCODEWORD Summary of this function goes here
%   Detailed explanation goes here
y_gf = H_gf * x_gf;

y = y_gf.x;

[n,m] = size(y)

correct = zeros(0,1);
for i = 1:n
    correct(i) = all(y(i,:)==0);
end

isCodeword = all(correct(:) == true);
end

