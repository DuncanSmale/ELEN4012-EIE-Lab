
clc; close all;
% this is the file used to test some of the generation used for the lab

file_prefix = "Python/models/";

file_suffix = ".h5";
matlabNetTypes = [InputTypes.LLR];
matlabNets = cell(numel(matlabNetTypes),1);
for i = 1:numel(matlabNetTypes)
    name = string(matlabNetTypes(i)) + '.mat';
    matlabNets{i} = load(name, 'net');
end
%models = ["LLR"];
labels = [];
nets = cell(numel(models),1);
for i = 1:numel(models)
    file = file_prefix + models(i) + file_suffix;
    nets{i} = importKerasNetwork(file);
end
%SNR = -20:1:20;

% seed = 1; % seeding the random number generation for recontruction
% rng(seed);
matrix = readmatrix("newMatrix.txt", "Delimiter", " ");
columns = readmatrix("columns.txt", "Delimiter", " ");
columns = columns + 1;
info_loc = columns(1:end/2);
parity_loc = columns(end/2+1:end);
H = matrix;
H2 = H(:, columns);

[temp, order] = sort(columns);
spH2 = sparse(H2);
decoderLDPC = comm.LDPCDecoder('ParityCheckMatrix',spH2);
encoder = comm.LDPCEncoder('ParityCheckMatrix',spH2);

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;
%%%%%%% make these either 1 or 0, 1 is use, 0 is do not use
use_keras = 0;
use_matlab = 0;
%%%%%%%
n_blocks = 1*10^5;
%%%%%%%
SNR = 2:0.5:8;

decoder = zeros(size(SNR));
keras = zeros(size(SNR,1), numel(nets));
matlab = zeros(size(SNR));
hard = zeros(size(SNR));
parfor  i = 1:numel(SNR)
    disp("SNR: " + SNR(i))
    errors_decoder = 0;
    errors_keras = zeros(1, numel(nets));
    errors_hard = 0;
    errors_matlab = 0;
    
    decoderLDPC = comm.LDPCDecoder('ParityCheckMatrix',spH2);
    encoder = comm.LDPCEncoder('ParityCheckMatrix',spH2);
    demodulator = comm.BPSKDemodulator;
    modulator = comm.BPSKModulator;
    tic
    for j = 1:n_blocks
        m = randi([0 1], 1, 100);
        c = GetCodeword(encoder, info_loc, parity_loc, m);
        c_mod = real(modulator(c));
        noise = GetNoise(size(c_mod), SNR(i));
        x = c_mod + noise;
        x_thresh = x;
        
        xhat = real(demodulator(x));
        votes = GetVotes(H, xhat);
%         snr_calc = snr(c_mod, x-c_mod);
        snr_calc = SNR(i);
        variance = (1/2)*10^(-SNR(i)/10);
        x = GetLLR(x, snr_calc);
        decoded_matlab = decoderLDPC(x([info_loc parity_loc]));
        if use_keras == 1
            for k = 1:numel(nets)
                % getting relevant dataset
                if contains(models(k), string(InputTypes.LLRVoteRange))
                    %note I made the order now (Naive,LLR,Votes) for all novel input schemes in Schemes.m
                    flipped = Schemes.processFlipFromVote(x_thresh,x, votes);
                    arr = [flipped; variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLRVoteRange)];
                    end
                    
                elseif contains(models(k), string(InputTypes.LLRMultVoteMultNaive))
                    %Also: Am I understanding 'Schemes.interpret_demod_bpsk(...)'
                    %correctly?
                    %My idea:
                    LLRMultVoteMultNaive = Schemes.processLLRMultVoteMultNaive(xhat,x,votes'); %Schemes.interpret_demod_bpsk(x_thresh,0)
                    arr = [LLRMultVoteMultNaive; variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLRMultVoteMultNaive)];
                    end
                elseif contains(models(k), string(InputTypes.LLRMultVote))
                    %My idea:
                    LLRMultVote = Schemes.processLLRMultVote(x,votes');
                    arr = [LLRMultVote; variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLRMultVote)];
                    end
                elseif contains(models(k), string(InputTypes.NaiveMultVote))
                    %My idea:
                    NaiveMultVote = Schemes.processNaiveMultVote(xhat,votes');
                    arr = [NaiveMultVote', variance];
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.NaiveMultVote)];
                    end
                elseif contains(models(k), string(InputTypes.LLRVote))
                    %My idea:
                    LLRVote = [x, votes];
                    arr = [LLRVote;variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLRVote)];
                    end
                elseif contains(models(k), string(InputTypes.NaiveVote))
                    %My idea:
                    NaiveVote = [Schemes.interpret_demod_bpsk(x_thresh,0), votes];
                    arr = [NaiveVote;variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.NaiveVote)];
                    end
                elseif contains(models(k), string(InputTypes.Vote))
                    %My idea:
                    arr = [votes,variance];
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.Vote)];
                    end
                elseif contains(models(k), string(InputTypes.LLR))
                    arr = [x; variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLR)];
                    end
                elseif contains(models(k), string(InputTypes.Naive))
                    %My idea:
                    arr = [xhat; variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.Naive)];
                    end
                end
                
                decoded_netork = predict(nets{k}, arr);
                testkeras_round = round(decoded_netork);
                testkeras_check = xor(testkeras_round, m);
                
                errors_keras(k) = errors_keras(k) + sum(testkeras_check);
            end
        end
        if use_matlab == 1
            network_matlab = predict(matlabNets{1}.net, [x; variance]');
            testmatlab_round = round(network_matlab);
            testmatlab_check = xor(testmatlab_round, m);
            errors_matlab = errors_matlab + sum(testmatlab_check);
        end
        
        testdecoder_check = xor(decoded_matlab', m);
        
        testhard_check = xor(xhat(info_loc)', m);
        errors_decoder = errors_decoder + sum(testdecoder_check);
        errors_hard = errors_hard + sum(testhard_check);
    end
    decoder(i) = errors_decoder;
    keras(i, :) = errors_keras;
    matlab(i) = errors_matlab;
    hard(i) = errors_hard;
    toc
end
decoder = (decoder/(n_blocks*200))';
keras = (keras/(n_blocks*200));
matlab = (matlab/(n_blocks*200))';
hard = (hard/(n_blocks*200))';

SNR = SNR';

newArr = [SNR, decoder, hard];
true_labels = ["decoder","hard"];
if use_keras == 1
    true_labels = [true_labels, labels];
    newArr = [newArr, keras];
end
if use_matlab == 1
    true_labels = [true_labels, "MATLAB"];
    newArr = [newArr, matlab];
end

tableVars = ["SNR", true_labels];
vars = cellstr(tableVars);
Decoders = array2table(newArr, 'VariableNames', vars)

%Optional Plotting
figure
semilogy(SNR, newArr(:, 2:end),'LineWidth', 2)
%Because 'keras' contains two models, include two legend items - see below
%Give proper names for legend when feel it's time

legend(true_labels);
xlabel("SNR");
ylabel("BER");

