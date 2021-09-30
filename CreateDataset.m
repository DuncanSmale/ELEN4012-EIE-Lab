function [dataset,messages] = CreateDataset(type, num_messages, H, percent_noisy, seed, SNR)
%CREATEDATASET creates a dataset based on given parameters
%   type:           InputTypes - Naive, LLR, Votes, LLRVotes
%   num_messages:   int - how many messages
%   H:              [N-K, N] double - Parity Check Matrix
%   percent noisy:  double (0-1) - how much of the dataset is noise
%   seed:           int - rng seed
%   SNR:            [] double - the SNR values to generate data for

percent_noisy = clamp(percent_noisy, 0, 1);
percent_noisy

B_gf = gf(H(:, 1:end/2));
A_gf = gf(H(:, end/2 + 1:end));
A_inv = inv(A_gf);

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

rng(seed);
messages = randi([0 1], num_messages, 100);

noisy_index = (1-percent_noisy) * num_messages; % how much of the data is noise
num_noisy = percent_noisy * num_messages;
noisy_index
num_noisy

%disp("Dataset:")
dataset = GetCodeword(A_inv, B_gf, messages);
correct = CheckCodeword(gf(H), dataset)
%disp("To be Noise-added (Right) Section:")
x = dataset(:, noisy_index+1:end);

num_per_noise = num_noisy/numel(SNR);
step = num_per_noise
index = 1;

for i = 1:size(x,2)
    x(:, i) = real(modulator(x(:, i)))';
end

if percent_noisy ~= 0
    for i = SNR
        % add noise to each codeword: "received" vector
        x(:, index:index+step-1) = awgn(x(:, index:index+step-1), i, 'measured');
    end
end
%disp("Received Nosiy x:")
x;
%[sizerow,sizecol] = size(x)
if type == InputTypes.Naive %% Naive dataset
    
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end

    if percent_noisy ~= 0
        dataset(:, noisy_index + 1: end) = x;
    end
    
elseif type == InputTypes.NaiveSyndrome %% NaiveSyndrome dataset
    
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end
    
    if percent_noisy ~= 0
        dataset(:, noisy_index + 1: end) = x;
    end
    size(dataset)
    size(H)
    syndromes = mod(H * dataset, 2);
    
    dataset = [dataset; syndromes];
        
elseif type == InputTypes.LLR %% LLR dataset 
    index = 1;
    for j = SNR
        x(:, index:index+step-1) = GetLLR(x(:, index:index+step-1), j);
        index = index + step;
    end
    
    dataset(:, noisy_index + 1: end) = x;
    
    % since we are using LLR in the whole data set we need to get the 
    % LLR of the non noisy sections
    for i = 1:noisy_index
        dataset(:, i) = real(modulator(dataset(:, i)))';
    end
    dataset(:, 1: noisy_index) = GetLLR(dataset(:, 1: noisy_index), 15);
    dataset = round(dataset, 3);

elseif type == InputTypes.Vote %% Vote dataset
    temp_votes = zeros(size(dataset));
    %demodulate noisy part
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end

    if percent_noisy ~= 0
        %Add the nosiy part to dataset
        dataset(:, noisy_index + 1: end) = x;
        %calculate the votes for non-noisy (should be all 0) and noisy parts
        for j = 1:size(dataset,2)
            temp_votes(:,j) = GetVotes(H,dataset(:,j));
        end
        %Add the votes to the dataset
        dataset = [dataset; temp_votes];
    end

elseif type == InputTypes.LLRVote %% LLR + Votes dataset
    %Copies of the original x and dataset need to be kept to work out votes
    x_votes = x;
    dataset_votes = dataset;

    %----------LLR Part----------
    index = 1;
    for j = SNR
        x(:, index:index+step-1) = GetLLR(x(:, index:index+step-1), j);
        index = index + step;
    end
    
    dataset(:, noisy_index + 1: end) = x;
    
    % since we are using LLR in the whole data set we need to get the 
    % LLR of the non noisy sections
    for i = 1:noisy_index
        dataset(:, i) = real(modulator(dataset(:, i)))';
    end
    dataset(:, 1: noisy_index) = GetLLR(dataset(:, 1: noisy_index), 0);
    dataset = round(dataset, 3);

    %----------Votes Part----------
    temp_votes = zeros(size(dataset_votes));
    %demodulate noisy part
    for j = 1:size(x_votes,2)
       x_votes(:, j) = real(demodulator(x_votes(:, j)))';
    end

    if percent_noisy ~= 0
        %Add the nosiy part to dataset
        dataset_votes(:, noisy_index + 1: end) = x_votes;
        %calculate the votes for non-noisy (should be all 0) and noisy parts
        for j = 1:size(dataset_votes,2)
            temp_votes(:,j) = GetVotes(H,dataset_votes(:,j));
        end
        %Add the votes to the dataset
        dataset = [dataset; temp_votes];
        %[rowds,colds] = size(dataset)
    end
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

