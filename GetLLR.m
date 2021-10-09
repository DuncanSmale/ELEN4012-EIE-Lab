function LLR = GetLLR(received, SNR)
% received:     received codeword
% SNR:          Signal to noise ration of the channel
% LLR:          reliability of each node as an LLR (log likelihood ratio)

variance = (1/2)*10^(-SNR/10);
LLR = (2/variance)*received;
end

