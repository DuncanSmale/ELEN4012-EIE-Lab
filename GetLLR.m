function LLR = GetLLR(received, SNR)
% received:     received codeword
% SNR:          Signal to noise ration of the channel
% LLR:          reliability of each node as an LLR (log likelihood ratio)

EbNo = 10^(SNR/10);
var_noise = 1/EbNo;
LLR = (2/var_noise)*received;
end

