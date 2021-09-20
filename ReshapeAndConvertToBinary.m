function H = ReshapeAndConvertToBinary(str,N,M)
Hm = reshape(str, [M, N]);
H = zeros(M,N);
for i = 1:N
    for j = 1:M
        if Hm(j,i) == '1'
            H(j,i) = 1;
        end
    end
end
end
