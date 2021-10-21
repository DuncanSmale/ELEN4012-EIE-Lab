function [dataset,messages] = CreateDataset(type, num_messages, encoder, info_loc, parity_loc,  H, seed, SNR, SNR_weights)
%CREATEDATASET creates a dataset based on given parameters
%   type:           InputTypes - Naive, LLR, Votes, LLRVotes
%   num_messages:   int - how many messages
%   H:              [N-K, N] double - Parity Check Matrix
%   percent noisy:  double (0-1) - how much of the dataset is noise
%   seed:           int - rng seed
%   SNR:            [] double - the SNR values to generate data for

% TODO add SNR to all data sets and fix the size of the inputs to 201

SNR_no_variance = 100;

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

rng(seed);
messages = randi([0 1], num_messages, 100);

% noisy_index
% num_noisy

%disp("Dataset:")
dataset = GetCodeword(encoder, info_loc, parity_loc, messages);
% correct = CheckCodeword(gf(H), dataset)
%disp("To be Noise-added (Right) Section:")
x = dataset;
if max(abs(mod(H*dataset,2)))>0
    error('Parity-check equations violated');
end

num_per_noise = SNR_weights * num_messages;

for i = 1:size(x,2)
    x(:, i) = real(modulator(x(:, i)))';
end
snrs = [];
index = 1;
for i = 1:numel(SNR)
    % add noise to each codeword: "received" vector
    step = num_per_noise(i);
    c_mod = x(:, index:index+step-1);
    dimens = [size(x,1), step];
    noise = GetNoise(dimens, SNR(i));
    x(:, index:index+step-1) = x(:, index:index+step-1) + noise;
    for j = index:index+step-1
        variance = (1/2)*10^(-SNR(i)/10);
        snrs = [snrs; variance];
    end
    index = index + step;
end

%disp("Received Nosiy x:")
x;
%[sizerow,sizecol] = size(x)
if type == InputTypes.Naive %% Naive dataset
    
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end

    dataset = x;
        
elseif type == InputTypes.LLR %% LLR dataset 
    index = 1;
    for j = 1:numel(SNR)
        step = num_per_noise(j);
        for k = index:index+step-1
%             actualSNR = snr(c_mod(:, k - index + 1), x(:, k)-c_mod(:, k - index + 1));
            x(:, k) = GetLLR(x(:, k), SNR(j));
        end
        
        index = index + step;
    end
    
    dataset = x;
    
    % since we are using LLR in the whole data set we need to get the 
    % LLR of the non noisy sections
    dataset = round(dataset, 3);

elseif type == InputTypes.Vote
    temp_votes = zeros(size(dataset));
    %demodulate noisy part
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end

    %Add the nosiy part to dataset
    dataset = x;
    %calculate the votes for non-noisy (should be all 0) and noisy parts
    for j = 1:size(dataset,2)
        temp_votes(:,j) = GetVotes(H,dataset(:,j));
    end
    %Replace dataset with only votes 
    dataset = temp_votes;

elseif type == InputTypes.NaiveMultVote %Quantized Naive * (Max Vote - Vote)
    temp_votes = zeros(size(dataset));
    %demodulate noisy part
    for j = 1:size(x,2)
        %x(:, j) = Schemes.interpret_demod_bpsk(real(demodulator(x(:, j)))',0); %Note this line is now decoded
        x(:, j) = real(demodulator(x(:, j)))';
    end

    %Add the nosiy part to dataset
    dataset = x;
    %calculate the votes for non-noisy (should be all 0) and noisy parts
    for j = 1:size(dataset,2)
        temp_votes(:,j) = GetVotes(H,dataset(:,j));
    end
    %max_votes = max(max(temp_votes));
    %Add the votes to the dataset
    %dataset = dataset .* (1./(1+temp_votes));
    dataset = Schemes.processNaiveMultVote(dataset,temp_votes);
    %%%dataset = dataset .* (max_votes-temp_votes);
    %Replace dataset with only votes 
    %dataset = dataset .* (1./(1+temp_votes));
    
elseif type == InputTypes.NaiveVote %% [Naive; Vote] dataset (was called 'Vote')
    temp_votes = zeros(size(dataset));
    %demodulate noisy part
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end

        %Add the nosiy part to dataset
        dataset = x;
        %calculate the votes for non-noisy (should be all 0) and noisy parts
        for j = 1:size(dataset,2)
            temp_votes(:,j) = GetVotes(H,dataset(:,j));
        end
        %Add the votes to the dataset
        dataset = [dataset; temp_votes];

elseif type == InputTypes.LLRVote %% [LLR; Votes] dataset
    %Copies of the original x and dataset need to be kept to work out votes
    x_votes = x;
    dataset_votes = dataset;

    %----------LLR Part----------
    index = 1;
    for j = 1:numel(SNR)
        step = num_per_noise(j);
        for k = index:index+step-1
%             actualSNR = snr(c_mod(:, k - index + 1), x(:, k)-c_mod(:, k - index + 1));
            x(:, k) = GetLLR(x(:, k), SNR(j));
        end
        
        index = index + step;
    end
    
    dataset = x;
    
    % since we are using LLR in the whole data set we need to get the 
    % LLR of the non noisy sections
    for i = 1:size(dataset,2)
        dataset(:, i) = real(modulator(dataset(:, i)))';
    end

    dataset = round(dataset, 3);

    %----------Votes Part----------
    temp_votes = zeros(size(dataset_votes));
    %demodulate noisy part
    for j = 1:size(x_votes,2)
       x_votes(:, j) = real(demodulator(x_votes(:, j)))';
    end

        %Add the nosiy part to dataset
        dataset_votes = x_votes;
        %calculate the votes for non-noisy (should be all 0) and noisy parts
        for j = 1:size(dataset_votes,2)
            temp_votes(:,j) = GetVotes(H,dataset_votes(:,j));
        end
        %Add the votes to the dataset
        dataset = [dataset; temp_votes];
        %[rowds,colds] = size(dataset)
elseif contains(string(type),string(InputTypes.LLRMultVote))
%elseif type == InputTypes.LLRMultVote %% LLR * (Max Votes - Votes)
    %%See many options in last line of elseif
    
    %Copies of the original x and dataset need to be kept to work out votes
    x_votes = x;
    dataset_votes = dataset;

    %----------LLR Part----------
    index = 1;
    for j = 1:numel(SNR)
        step = num_per_noise(j);
        for k = index:index+step-1
%             actualSNR = snr(c_mod(:, k - index + 1), x(:, k)-c_mod(:, k - index + 1));
            x(:, k) = GetLLR(x(:, k), SNR(j));
        end
        
        index = index + step;
    end
    
    dataset = x;
    
    % since we are using LLR in the whole data set we need to get the 
    % LLR of the non noisy sections
    dataset = round(dataset, 3);

    %----------Votes Part----------
    temp_votes = zeros(size(dataset_votes));
    %demodulate noisy part
    for j = 1:size(x_votes,2)
       x_votes(:, j) = real(demodulator(x_votes(:, j)))';
    end
    %----------x_naive----------
    if(contains(string(type),string(InputTypes.LLRMultVoteMultNaive)))
        x_naive = x_votes;
%         for j = 1:size(x_naive,2)
%            %x_naive(:, j) = Schemes.interpret_demod_bpsk(real(demodulator(x_naive(:, j)))',0);
%            %x_naive(:, j) = real(demodulator(x_naive(:, j)))';
%         end
    end
    %------------------------------
    %Add the nosiy part to dataset
    dataset_votes = x_votes;
    %calculate the votes for non-noisy (should be all 0) and noisy parts
    for j = 1:size(dataset_votes,2)
        temp_votes(:,j) = GetVotes(H,dataset_votes(:,j));
    end
    %max_votes = max(max(temp_votes));
    %Other possibilities for input schemes
    %Add the votes to the dataset
    %dataset = dataset .* (1./(1+temp_votes));
    %dataset = dataset + (1/1).*(max_votes-temp_votes);
    %dataset = dataset + (max_votes-temp_votes);
    %dataset = dataset + (max_votes-temp_votes) + x_votes;
    %dataset = [x_votes ; dataset .* (max_votes-temp_votes)];

    if(contains(string(type),string(InputTypes.LLRMultVoteMultNaive)))
        %dataset = dataset .* (max_votes-temp_votes) .* x_naive;
        dataset = Schemes.processLLRMultVoteMultNaive(x_naive,dataset,temp_votes);
    else
        %dataset = dataset .* (max_votes-temp_votes);
        dataset = Schemes.processLLRMultVote(dataset,temp_votes);
    end
    %[rowds,colds] = size(dataset)
elseif type == InputTypes.LLRVoteRange
    x_thresh = x;
    x_LLR = x;
    %demodulate noisy part
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end
    
    x_votes = x;
    dataset_votes = dataset;
    index = 1;
    %----------Votes Part----------
    temp_votes = zeros(size(dataset_votes));
    
    %Add the nosiy part to dataset
    dataset_votes = x_votes;
    %calculate the votes for non-noisy (should be all 0) and noisy parts
    for j = 1:size(dataset_votes,2)
        temp_votes(:,j) = GetVotes(H,dataset_votes(:,j));
    end
    %----------LLR Part----------
    index = 1;
    for j = 1:numel(SNR)
        step = num_per_noise(j);
        for k = index:index+step-1
            x_LLR(:, k) = GetLLR(x_LLR(:, k), SNR(j));
            vec = x_thresh(:, k);
            vec = Schemes.processFlipFromVote(vec, x_LLR(:, k), temp_votes(:,k));
            dataset(:, k) = vec;
        end
        
        index = index + step;
    end
    
    dataset = round(dataset, 3);
end

dataset = dataset';
snrs = round(snrs, 5);
size(snrs);
dataset = [dataset snrs];
% shuffle rows to make sure that the noisy data is mixed with the normal
% data, this will ensure minimal data bias
% random_order = randperm(num_messages);
% dataset = dataset(random_order, :);
% messages = messages(random_order,:);
end


