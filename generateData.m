%%% This script is used to generate the required data sets for training %%%
%%% Change the parameters to alter the dataset given back %%%
% TODO add SNR to all data sets and fix the size of the inputs to 201
clc
load H.mat H_rev

% InputTypes are: Naive, NaiveSyndrome, LLR, Vote, LLRVote
type = InputTypes.LLR; 

% how many messages/codewords to generate
num_messages = 20000;

% make this between 0 and 1
percent_noisy = 0.9;

% rng seed to use
seed = 1;

% Try make the total elements divisible by 10
% range of SNR values to use in the noisy portion of the dataset
% SNR = 0:0.2:4.8; 
SNR = 0:2:6;

[dataset, messages] = CreateDataset(type, num_messages, H_rev, percent_noisy, seed, SNR);

% dont change
full_percent = num2str(percent_noisy * 100, '%u');
datapoints = NumToString(num_messages);
numSNR = "_" + num2str(SNR(1)) + "_" + num2str(SNR(end)) + "SNR";

% change to desired name of text files
% this function creates the folder of name LLR, change this to the string
% version of string(type) i.e if its votes change LLR to votes
mkdir(string(type))
message_name = string(type) + "/" + datapoints + numSNR + full_percent + "messages" + string(type) + ".txt";
dataset_name = string(type) + "/" + datapoints + numSNR + full_percent + "data" + string(type) + ".txt";

writematrix(dataset, dataset_name);
writematrix(messages, message_name);

function num_string = NumToString(num)
num_string = num2str(num);
if mod(num, 1e6) ~= num
    format = "%u";
    if mod(num, 1e6) ~= 0
        format = "%.1f";
    end
    num_string = num2str(num/1e6, format) + "M";
elseif mod(num, 1e3) ~= num
    num_string = num2str(num/1e3, "%u") + "K";
end
end