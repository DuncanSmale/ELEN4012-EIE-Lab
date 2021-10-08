%%% This script is used to generate the required data sets for training %%%
%%% Change the parameters to alter the dataset given back %%%
% TODO add SNR to all data sets and fix the size of the inputs to 201
clc

H = readmatrix("newMatrix.txt", "Delimiter", " ");
columns = readmatrix("columns.txt", "Delimiter", " ");
columns = columns + 1;
info_loc = columns(1:end/2);
parity_loc = columns(end/2+1:end);
H2 = H(:, columns);

[temp, order] = sort(columns);
spH2 = sparse(H2);
encoder = comm.LDPCEncoder('ParityCheckMatrix',spH2);

% InputTypes are: Naive, NaiveSyndrome, LLR, Vote, NaiveVote, LLRVote, 
% NaiveMultVote, LLRMultVote, LLRMultVoteMultNaive, LLRVoteRange
type = InputTypes.LLR; 

% how many messages/codewords to generate
num_messages = 10;

size_validation = 0.2;

% make this between 0 and 1
percent_noisy = 1;

% rng seed to use
seed = 1;

% Try make the total elements divisible by 10
% range of SNR values to use in the noisy portion of the dataset
% SNR = 0:0.2:4.8; 
SNR = [20:2.5:30];

[datasetTrain, messagesTrain] = CreateDataset(type, num_messages, encoder, info_loc, parity_loc, H, percent_noisy, seed, SNR);

[datasetTest, messagesTest] = CreateDataset(type, num_messages * size_validation,encoder, info_loc, parity_loc, H, percent_noisy, seed+1, SNR);

% dont change
full_percent = num2str(percent_noisy * 100, '%u');
datapoints = NumToString(num_messages);
numSNR = "_" + num2str(SNR(1)) + "_" + num2str(SNR(end)) + "SNR";

% change to desired name of text files
% this function creates the folder of name LLR, change this to the string
% version of string(type) i.e if its votes change LLR to votes
mkdir(string(type))
message_name_train = string(type) + "/" + datapoints + numSNR + full_percent + "messagesTRAIN" + string(type) + ".txt";
dataset_name_train = string(type) + "/" + datapoints + numSNR + full_percent + "dataTRAIN" + string(type) + ".txt";

message_name_test = string(type) + "/" + datapoints + numSNR + full_percent + "messagesTEST" + string(type) + ".txt";
dataset_name_test = string(type) + "/" + datapoints + numSNR + full_percent + "dataTEST" + string(type) + ".txt";

writematrix(datasetTrain, dataset_name_train);
writematrix(messagesTrain, message_name_train);

writematrix(datasetTest, dataset_name_test);
writematrix(messagesTest, message_name_test);

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