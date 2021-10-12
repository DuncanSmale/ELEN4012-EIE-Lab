clc

N = 200;
M = 100;
K = 100;
Eb = 1;

H = readmatrix("newMatrix.txt", "Delimiter", " ");
columns = readmatrix("columns.txt", "Delimiter", " ");
columns = columns + 1;
info_loc = columns(1:end/2);
parity_loc = columns(end/2+1:end);
H2 = H(:, columns);

% writematrix(H2, 'parityCheck.txt', "Delimiter", " ")

[temp, order] = sort(columns);
spH2 = sparse(H2);
encoder = comm.LDPCEncoder('ParityCheckMatrix',spH2);

% seed = 1; % seeding the random number generation for recontruction
% rng(seed);
num_messages = 1;
m = randi([0 1], num_messages, 100);

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
SNR = [0];

for i = SNR
    noise = GetNoise([200 1], i);

    % var_calc = var(noise);

    c = GetCodeword(encoder, info_loc, parity_loc, m);
    c_mod = 1 - 2 * c;
    x = c_mod + noise;

    vari = (1/2)*10^(-i/10)
    sigma = sqrt(vari)
    check = mod(H2*c([info_loc parity_loc]),2)
    all(check == 0)

    LLR = GetLLR(x, i);
    % LLR = GetLLR(x, SNR_func);

    total = [c x noise LLR];
    disp(total(1:10, :));
end

