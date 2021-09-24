function [dataset,messages] = CreateDataset(type, num_messages, H, percent_noisy, seed, SNR)
%CREATEDATASET creates a dataset based on given parameters
%   type:           InputTypes - Naive, LLR, Votes, LLRVotes
%   num_messages:   int - how many messages
%   H:              [N-K, N] double - Parity Check Matrix
%   percent noisy:  double (0-1) - how much of the dataset is noise
%   seed:           int - rng seed
%   SNR:            [] double - the SNR values to generate data for

percent_noisy = clamp(percent_noisy, 0, 1);

B_gf = gf(H(:, 1:end/2));
A_gf = gf(H(:, end/2 + 1:end));
A_inv = inv(A_gf);

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

rng(seed);
messages = randi([0 1], num_messages, 100);

noisy_index = (1-percent_noisy); % how much of the data is noise
num_noisy = percent_noisy * num_messages;

dataset = GetCodeword(A_inv, B_gf, messages);
x = dataset(:, end*noisy_index+1:end);

num_per_noise = num_noisy/size(SNR,2);
step = num_per_noise;
index = 1;

for i = 1:size(x,2)
    x(:, i) = real(modulator(x(:, i)))';
end

for i = SNR
    % add noise to each codeword: "received" vector
    x(:, index:index+step-1) = awgn(x(:, index:index+step-1), i, 'measured');
    index = index + step;
end


if type == InputTypes.Naive %% Naive dataset
    
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end
    
    dataset(:, num_noisy + 1: end) = x;

elseif type == InputTypes.LLR %% LLR dataset 
    index = 1;
    for j = SNR
        x(:, index:index+step-1) = GetLLR(x(:, index:index+step-1), j);
        index = index + step;
    end
    
    dataset(:, num_noisy + 1: end) = x;
    
    % since we are using LLR in the whole data set we need to get the 
    % LLR of the non noisy sections
    for i = 1:num_noisy
        dataset(:, i) = real(modulator(dataset(:, i)))';
    end
    dataset(:, 1: num_noisy) = GetLLR(dataset(:, 1: num_noisy), 0);

elseif type == InputTypes.Votes %% Vote dataset
    
elseif type == InputTypes.LLRVotes %% LLR + Votes dataset
    
end

dataset = dataset';

% shuffle rows to make sure that the noisy data is mixed with the normal
% data, this will ensure minimal data bias
random_order = randperm(num_messages);
dataset = dataset(random_order, :);
messages = messages(random_order,:);
end

function y = clamp(x,bl,bu)
  % return bounded value clipped between bl and bu
  y=min(max(x,bl),bu);
end

