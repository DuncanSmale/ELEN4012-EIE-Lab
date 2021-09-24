%%% This script is used to generate the required data sets for training %%%
%%% Change the parameters to alter the dataset given back %%%

clc
load H.mat H_rev

% InputTypes are: Naive, LLR, Vote, LLRVote
type = InputTypes.LLR; 

% how many messages/codewords to generate
num_messages = 1000;

% make this between 0 and 1
percent_noisy = 0.5;

% rng seed to use
seed = 1;

% Try make the total elements divisible by 10
% range of SNR values to use in the noisy portion of the dataset
SNR = -5:0.2:4.8; 

[dataset, messages] = CreateDataset(type, num_messages, H_rev, percent_noisy, seed, SNR);

% change to desired name of text files
message_name = "testMessageLLR.txt";
dataset_name = "testDatasetLLR.txt";

writematrix(dataset, dataset_name);
writematrix(messages, message_name);