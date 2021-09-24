function [] = Simulate(SNR_start, SNR_end, SNR_step, Num_messages)
%SIMULATE a simulation of the LDPC encoding and decoding through AWGN channel
%   Detailed explanation goes here

% load parity check matrix
load H.mat H_rev
H_gf = gf(H_rev); % use to verify codewords
B_gf = gf(H_rev(:, 1:end/2));
A_gf = gf(H_rev(:, end/2 + 1:end));
A_inv = inv(A_gf);

A = double(A_gf.x);
B = double(B_gf.x);

[row, col] = find(H_rev);
I = [row col];
index = sparse(I(:,1),I(:,2),1);
encoder = comm.LDPCEncoder('ParityCheckMatrix',index);

% if the received bit is positive it is more likely to be a 0
decoder = comm.LDPCDecoder('ParityCheckMatrix',index);

bpsk_mod = comm.BPSKModulator;
bpsk_demod = comm.BPSKDemodulator;

[N, N_K] = size(H);
K = N - N_K;
seed = 1; % seeding the random number generation for recontruction
rng(seed);
m_full = randi([0 1], Num_messages, K); % create vector of messages
Noise = SNR_start:SNR_step:SNR_end;
for i = 1:size(Noise(2))
    % encode each message
    x = GetCodeword(A_inv, B_gf, m_full);
    % modulate each codeword
    mod_codeword = real(bpsk_mod(x))';
    % add noise to each codeword: "received" vector
    received = awgn(mod_codeword, Noise(i), 'measured');
    % reliability
    LLR = GetLLR(received, Noise(i));
    % demodulate each "received" vector
    % calculate votes
    % parity check bits
    % decode using MATLAB
    decoded_matlab = decoder(LLR');
    decoded_matlab = double(decoded_matlab)';
    % decode using Naive
    % decode using reliability
    % decode using vote
    % decode using reliability and vote
    % pass values to relevant deep learning schemes, store result
end
% generate BER
% plot BER vs SNR
end

