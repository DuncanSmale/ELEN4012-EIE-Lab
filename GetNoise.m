function noise = GetNoise(dimensions, SNR)
%ADDNOISE Summary of this function goes here
%   Detailed explanation goes here
%   variance = r*Eb*10^(-SNR/10)
%   Eb = 1
variance = (1/2)*10^(-SNR/10);
sigma = sqrt(variance);
noise = sigma * randn(dimensions);
end

