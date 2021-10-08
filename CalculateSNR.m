function SNR = CalculateSNR(received)
%CALCULATESNR Summary of this function goes here
%   Detailed explanation goes here
%   Power of sent always = 1
L=length(received);
P_y=(sum(abs(received).^2))/L;
P_n = P_y-1;

SNR = 10*log10(1/P_n);
end

