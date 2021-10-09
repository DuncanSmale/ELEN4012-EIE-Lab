
clc; close all;
% this is the file used to test some of the generation used for the lab

file_prefix = "Python/models/";
file_suffix = "10K_0_6SNR100H3tanh.h5";
models = ["Naive", "LLR", "Vote", "NaiveMultVote",...
    "LLRMultVote", "LLRMultVoteMultNaive", "LLRVoteRange"];
%models = ["LLRMultVote1K_0_6SNR100H2tanh.h5","LLR1K_0_6SNR100H2tanh.h5"];
% file_suffix = "10K_20_30SNR100H8tanh.h5";

%file_suffix = ".h5";
% models = ["Naive", "LLR", "Vote", "NaiveMultVote",...
%     "LLRMultVote", "LLRMultVoteMultNaive", "LLRVoteRange"];
%models = ["LLR"];
labels = [];
nets = cell(numel(models),1);
for i = 1:numel(models)
    file = file_prefix + models(i) + file_suffix;
    nets{i} = importKerasNetwork(file);
end
SNR = -2:10;

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

n_blocks = 1000;
decoder = zeros(size(SNR));
keras = zeros(size(SNR,1), numel(nets));
hard = zeros(size(SNR));
for  i = 1:numel(SNR)
    SNR(i)
    errors_decoder = 0;
    errors_keras = zeros(1, numel(nets));
    errors_hard = 0;
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
        
        testdecoder_check = xor(decoded_matlab', m);
        
        testhard_check = xor(xhat(info_loc)', m);
        % check_decoder = all(testdecoder_check(:) == 0)
        % check_keras = all(testkeras_check(:) == 0)
        errors_decoder = errors_decoder + sum(testdecoder_check);
        errors_hard = errors_hard + sum(testhard_check);
    end
    decoder(i) = errors_decoder;
    keras(i, :) = errors_keras;
    hard(i) = errors_hard;
end
labels = ["decoder",labels,"hard"];
SNR = SNR';
decoder = (decoder/(n_blocks*200))';
keras = (keras/(n_blocks*200));
hard = (hard/(n_blocks*200))';
tableVars = ["SNR", labels];
vars = cellstr(tableVars);
newArr = [SNR, decoder, keras, hard];
Decoders = array2table(newArr, 'VariableNames', vars)

%Optional Plotting
figure
hold on
semilogy(SNR, decoder,'LineWidth',2)
semilogy(SNR, keras,'LineWidth',2)
semilogy(SNR, hard,'LineWidth',2)
hold off
%Because 'keras' contains two models, include two legend items - see below
%Give proper names for legend when feel it's time

legend(labels);
xlabel("SNR");
ylabel("BER");

