%%% This script is used to generate the required data sets for training %%%
%%% Change the parameters to alter the dataset given back %%%

clc
load H.mat H_rev

% InputTypes are: Naive, LLR, Vote, LLRVote
type = InputTypes.LLR; 

% how many messages/codewords to generate
num_messages = 2000000;

% make this between 0 and 1
percent_noisy = 0.75;

% rng seed to use
seed = 1;

% Try make the total elements divisible by 10
% range of SNR values to use in the noisy portion of the dataset
% SNR = 0:0.2:4.8; 
SNR = 0;

[dataset, messages] = CreateDataset(type, num_messages, H_rev, percent_noisy, seed, SNR);

% change to desired name of text files
% this function creates the folder of name LLR, change this to the string
% version of string(type) i.e if its votes change LLR to votes
% mkdir LLR 
message_name = string(type) + "/2M75messages" + string(type) + ".txt";
dataset_name = string(type) + "/2M75data" + string(type) + ".txt";

writematrix(dataset, dataset_name);
writematrix(messages, message_name);