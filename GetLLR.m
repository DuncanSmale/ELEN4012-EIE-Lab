function LLR = GetLLR(received, SNR)
% received:     received codeword
% SNR:          Signal to noise ration of the channel
% LLR:          reliability of each node as an LLR (log likelihood ratio)
SNR_dec = 10^(SNR/10);
N0_uncoded = 1/SNR_dec; % uncoded 
N0 = N0_uncoded/(1/2); % half rate code
LLR = (2/N0)*received;
end

