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
% Naive, LLR, Vote, NaiveMultVote, LLRMultVote, LLRMultVoteMultNaive, LLRVoteRange
% allTypes = [InputTypes.Naive, InputTypes.LLR, InputTypes.Vote, ...
%     InputTypes.NaiveMultVote, InputTypes.LLRMultVote, ... 
%     InputTypes.LLRMultVoteMultNaive, InputTypes.LLRVoteRange];
allTypes = [InputTypes.LLRMultVoteMultNaive];
for type = allTypes
    disp(type)
    % how many messages/codewords to generate
    num_messages = 1000;

    size_validation = 0.1;


    % rng seed to use
    seed = 1;

    % Try make the total elements divisible by 10
    % range of SNR values to use in the noisy portion of the dataset
    % SNR = 0:0.2:4.8; 
    %SNR = 10:1:29;
    SNR = [2 4 6 8]; 
    SNR_weights = [0.1 0.2 0.3 0.4];
    if sum(SNR_weights) ~= 1
       warning("SNR Weights do not sum to 1") 
       return
    end

    [datasetTrain, messagesTrain] = CreateDataset(type, num_messages,...
        encoder, info_loc, parity_loc, H, seed, SNR, SNR_weights);
    
    total_validation = num_messages * size_validation;
    
    [datasetTest, messagesTest] = CreateDataset(type, total_validation, ...
        encoder, info_loc, parity_loc, H, seed*10, SNR, SNR_weights);

    % dont change
    datapoints = NumToString(num_messages);
    numSNR = "_" + num2str(SNR(1)) + "_" + num2str(SNR(end)) + "SNR";

    % change to desired name of text files
    % this function creates the folder of name LLR, change this to the string
    % version of string(type) i.e if its votes change LLR to votes
    if ~exist(string(type), 'dir')
        mkdir(string(type))
    end

    message_name_train = string(type) + "/" + datapoints + numSNR + "messagesTRAIN" + string(type) + ".txt";
    dataset_name_train = string(type) + "/" + datapoints + numSNR + "dataTRAIN" + string(type) + ".txt";

    message_name_test = string(type) + "/" + datapoints + numSNR + "messagesTEST" + string(type) + ".txt";
    dataset_name_test = string(type) + "/" + datapoints + numSNR + "dataTEST" + string(type) + ".txt";

    writematrix(datasetTrain, dataset_name_train);
    writematrix(messagesTrain, message_name_train);

    writematrix(datasetTest, dataset_name_test);
    writematrix(messagesTest, message_name_test);
end

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