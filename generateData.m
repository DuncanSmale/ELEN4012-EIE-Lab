clc
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

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

seed = 1;
rng(seed);
num_messages = 10000;
messages = randi([0 1], num_messages, 100);

data = GetCodeword(A_inv, B_gf, messages);

x = data(:, (end/2)+1:end);

num_noisy = 0.5 * num_messages;

Noise = -5:0.2:4.8;

num_per_noise = num_noisy/size(Noise,2)

index = 1;
step = 10;


for i = 1:size(x,2)
    x(:, i) = real(modulator(x(:, i)))';
end

for i = Noise
    % add noise to each codeword: "received" vector
    x(:, index:index+step-1) = awgn(x(:, index:index+step-1), i, 'measured');
    index = index + step;
end

for i = 1:size(x,2)
    x(:, i) = real(demodulator(x(:, i)))';
end

x_test = data(:, (end/2)+1:end);

check = xor(x, x_test);

sum(check(:));

data(:, num_noisy + 1: end) = x;

data = data';

writematrix(data);
writematrix(messages);