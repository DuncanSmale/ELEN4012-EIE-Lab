function noise = GetNoise(dimensions, SNR)
%ADDNOISE Summary of this function goes here
%   Detailed explanation goes here
EbNo = 10^(SNR/10);
sigma = sqrt(1/(EbNo));
noise = sigma * randn(dimensions);
end

