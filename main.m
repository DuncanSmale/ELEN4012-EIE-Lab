clc
% this is the file used to test some of the generation used for the lab

LLRModel = 'Python/models/LLR10K_0_6SNR100H2tanh.h5';
LLRNet = importKerasNetwork(LLRModel)
SNR = 0;

load H.mat H_rev
H = H_rev;

% seed = 1; % seeding the random number generation for recontruction
% rng(seed);
num_messages = 1;
m = randi([0 1], num_messages, 100);
m_gf = gf(m);

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

A_inv = inv(A_gf);

[row, col] = find(H_rev);
I = [row col];
index = sparse(I(:,1),I(:,2),1);
decoder = comm.LDPCDecoder('ParityCheckMatrix',index);

c = GetCodeword(A_inv, B_gf, m);
c_mod = real(modulator(c));
noise = GetNoise(size(c_mod), SNR);
x = c_mod + noise;
xhat = real(demodulator(x));
% synd = mod(H * xhat,2);
% naivesynd = [xhat; synd]';
snr_calc = snr(c_mod, x-c_mod)
x = GetLLR(x, snr_calc);
decoded_matlab = decoder(x);

decoded_netork = predict(LLRNet, [x; snr_calc]');
testkeras_round = round(decoded_netork);

testdecoder_check = xor(decoded_matlab', m);
testkeras_check = xor(testkeras_round, m);
testhard_check = xor(xhat(1:100)', m);
% check_decoder = all(testdecoder_check(:) == 0)
% check_keras = all(testkeras_check(:) == 0)
num_errors_decoder = sum(testdecoder_check)
num_errors_keras = sum(testkeras_check)
num_errors_hard = sum(testhard_check)