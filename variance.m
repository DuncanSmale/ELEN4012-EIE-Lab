clc

N = 200;
M = 100;
K = 100;
Eb = 1;

load H.mat H_rev
H = H_rev;

B_gf = gf(H(:, 1:end/2));
A_gf = gf(H(:, end/2 + 1:end));

A = double(A_gf.x);
B = double(B_gf.x);

A_inv = inv(A_gf);

[row, col] = find(H_rev);
I = [row col];
index = sparse(I(:,1),I(:,2),1);
decoder = comm.LDPCDecoder('ParityCheckMatrix',index);

% seed = 1; % seeding the random number generation for recontruction
% rng(seed);
num_messages = 1;
m = randi([0 1], num_messages, 100);
m_gf = gf(m);

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

% x = awgn(c_mod, SNR, 'measured');
% xhat = (x < 0);
% L=length(x);
% P_y = (norm(x)^2)/L
% n = x - c_mod;
% P_n = (norm(n)^2)/L;
% P_ycal = P_n + 1
% SNR_func = snr(c_mod, x-c_mod)

% sigp = 10*log10(norm(x,2)^2/numel(x));
% add noise to each codeword: "received" vector
% noisep_db = sigp-SNR;
% noisep = 10^(noisep_db/10);
SNR = 1;
noise = GetNoise(size(c_mod), SNR);
% var_calc = var(noise);
x = c_mod + noise;
SNR_func1 = snr(c_mod, x-c_mod)
c = GetCodeword(A_inv, B_gf, m);

c_mod = 1 - 2 * c;

Eb = 1;
EbNo = 10^(SNR_func1/10)
var_noise = 1/EbNo

LLR = (2/var_noise)*x;
% LLR = GetLLR(x, SNR_func);

total = [c x LLR];
disp(total(1:10, :))

