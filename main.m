clc
% this is the file used to test some of the generation used for the lab
N = 200;
M = 100;
K = 100;
% save("H.mat", "H_rev", "H");
% load H.mat H
% load messages.mat messages
% 
% messages = messages';
% writematrix(messages);

readmatrix('messages.txt');

% this for loop helped generate a non singular A matrix 
% for i = 1:1000
%     cols = randi([1 200], 1, 2);
%     h = H(:,cols(1));
%     H(:,cols(1)) = H(:,cols(2));
%     H(:,cols(2)) = h;
%     A = gf(H(:, 1:end/2));
%     if det(A) ~= 0
%         break
%     end
% end

A_gf = gf(H(:, 1:end/2));
B_gf = gf(H(:, end/2 + 1:end));

A = double(A_gf.x);
B = double(B_gf.x);

H_rev = [B A];
[row, col] = find(H_rev);
I = [row col];
index = sparse(I(:,1),I(:,2),1);
encoder = comm.LDPCEncoder('ParityCheckMatrix',index);

seed = 1; % seeding the random number generation for recontruction
rng(seed);
num_messages = 1;
m = randi([0 1], num_messages, 100);
m_gf = gf(m);

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

A_inv = inv(A_gf);

c = A_inv * B_gf * m';
x_gf = [m c']';
x = double(x_gf.x);
c_enc = encoder(m');

LogicalStr = {'false', 'true'};
check_arr = xor(x, c_enc);
check = all(check_arr(:)==0);

% checking mathematical encoding vs MATLAB encoding
fprintf("Encoder same as math: %s\n", LogicalStr{check + 1});

% as SNR increases check_dec will = 1, as message decoded = message sent
SNR = 10;
mod_codeword = real(bpsk(c_enc))';
noise_codeword = awgn(mod_codeword, SNR,'measured');

message = decoder(noise_codeword');
message = double(message)';

check_decarr = xor(message, m);
sum(check_decarr)
check_dec = all(check_decarr(:)==0)

H_gf = gf(H);

y_gf = H_gf * x_gf;

y = y_gf.x;

correct = zeros(0,1);
for i = 1:num_messages
    correct(i) = all(y(i,:)==0);
end

% all(correct(:) == true)