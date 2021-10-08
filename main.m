
clc
% this is the file used to test some of the generation used for the lab

file_prefix = "Python/models/";
file_suffix = "10K_0_6SNR100H3tanh.h5";
models = ["Naive", "LLR", "Vote", "NaiveMultVote",...
    "LLRMultVote", "LLRMultVoteMultNaive", "LLRVoteRange"];
%models = ["LLRMultVote1K_0_6SNR100H2tanh.h5","LLR1K_0_6SNR100H2tanh.h5"];
labels = [];
nets = cell(numel(models),1);
for i = 1:numel(models)
    file = file_prefix + models(i) + file_suffix;
    nets{i} = importKerasNetwork(file);
end
SNR = [0];

load H.mat H_rev
H = H_rev;

% seed = 1; % seeding the random number generation for recontruction
% rng(seed);
num_messages = 1;
m = randi([0 1], num_messages, 100);
m_gf = gf(m);

modulator = comm.BPSKModulator;
demodulator = comm.BPSKDemodulator;

B_gf = gf(H(:, 1:end/2));
A_gf = gf(H(:, end/2 + 1:end));
A_inv = inv(A_gf);

[row, col] = find(H_rev);
I = [row col];
index = sparse(I(:,1),I(:,2),1);
decoderLDPC = comm.LDPCDecoder('ParityCheckMatrix',index);
n_blocks = 1;
decoder = zeros(size(SNR));
keras = zeros(size(SNR,1), numel(nets));
hard = zeros(size(SNR));
for  i = 1:numel(SNR)
    errors_decoder = 0;
    errors_keras = zeros(1, numel(nets));
    errors_hard = 0;
    for j = 1:n_blocks
        c = GetCodeword(A_inv, B_gf, m);
        c_mod = real(modulator(c));
        noise = GetNoise(size(c_mod), SNR(i));
        x = c_mod + noise;
        x_thresh = x;
        xhat = real(demodulator(x));
        votes = GetVotes(H, xhat);
        snr_calc = snr(c_mod, x-c_mod);
        x = GetLLR(x, snr_calc);
        decoded_matlab = decoderLDPC(x);
        for k = 1:numel(nets)
            % getting relevant dataset
            models(k)
            if contains(models(k), string(InputTypes.LLRVoteRange))
                %note I made the order now (Naive,LLR,Votes) for all novel input schemes in Schemes.m
                flipped = Schemes.processFlipFromVote(x_thresh,x, votes)
                arr = [flipped; snr_calc]';
                labels = [labels, string(InputTypes.LLRVoteRange)];
                
            elseif contains(models(k), string(InputTypes.LLRMultVoteMultNaive))
                %Also: Am I understanding 'Schemes.interpret_demod_bpsk(...)'
                %correctly?
                %My idea:
                LLRMultVoteMultNaive = Schemes.processLLRMultVoteMultNaive(Schemes.interpret_demod_bpsk(x_thresh,0),x,votes')
                arr = [LLRMultVoteMultNaive; snr_calc]';
                labels = [labels, string(InputTypes.LLRMultVoteMultNaive)];
            elseif contains(models(k), string(InputTypes.LLRMultVote))
                %My idea:
                LLRMultVote = Schemes.processLLRMultVote(x,votes')
                arr = [LLRMultVote; snr_calc]';
                labels = [labels, string(InputTypes.LLRMultVote)];
                
            elseif contains(models(k), string(InputTypes.NaiveMultVote))
                %My idea:
                NaiveMultVote = Schemes.processNaiveMultVote(Schemes.interpret_demod_bpsk(x_thresh',0),votes)
                arr = [NaiveMultVote, snr_calc];
                labels = [labels, string(InputTypes.NaiveMultVote)];
                
            elseif contains(models(k), string(InputTypes.LLRVote))
                %My idea:
                LLRVote = [x, votes]
                arr = [LLRVote;snr_calc]';
                labels = [labels, string(InputTypes.LLRVote)];
                
            elseif contains(models(k), string(InputTypes.NaiveVote))
                %My idea:
                NaiveVote = [Schemes.interpret_demod_bpsk(x_thresh,0), votes]
                arr = [NaiveVote;snr_calc]';
                labels = [labels, string(InputTypes.NaiveVote)];
                
            elseif contains(models(k), string(InputTypes.Vote))
                %My idea:
                arr = [votes,snr_calc];
                labels = [labels, string(InputTypes.Vote)];
                
            elseif contains(models(k), string(InputTypes.LLR))
                arr = [x; snr_calc]';
                labels = [labels, string(InputTypes.LLR)];
                
            elseif contains(models(k), string(InputTypes.Naive))
                %My idea:
                arr = [x_thresh; snr_calc]'
                labels = [labels, string(InputTypes.Naive)];
                
            end
            decoded_netork = predict(nets{k}, arr);
            testkeras_round = round(decoded_netork);
            testkeras_check = xor(testkeras_round, m);
            errors_keras(k) = errors_keras(k) + sum(testkeras_check);
        end
        
        testdecoder_check = xor(decoded_matlab', m);
        
        testhard_check = xor(xhat(1:100)', m);
        % check_decoder = all(testdecoder_check(:) == 0)
        % check_keras = all(testkeras_check(:) == 0)
        errors_decoder = errors_decoder + sum(testdecoder_check);
        errors_hard = errors_hard + sum(testhard_check);
    end
    decoder(i) = errors_decoder;
    keras(i, :) = errors_keras;
    hard(i) = errors_hard;
end
SNR = SNR';
decoder = (decoder/(n_blocks*200))';
keras = (keras/(n_blocks*200));
hard = (hard/(n_blocks*200))';
Decoders = table(SNR, decoder, keras, hard)

%Optional Plotting
figure
hold on
semilogy(decoder)
semilogy(keras)
semilogy(hard)
hold off
%Because 'keras' contains two models, include two legend items - see below
%Give proper names for legend when feel it's time
labels = ["decoder",labels,"hard"];
legend(labels);
xlabel("SNR");
ylabel("BER");

