
clc; close all;
% this is the file used to test some of the generation used for the lab
tic
file_prefix = "Python/models/";
file_suffix = ".h5";

models1 = ["TEST6Naive", "TEST6LLR", "TEST6NaiveMultVote",...
    "TEST6LLRMultVote", "TEST6LLRVoteRange"];
models2 = ["TEST6NaiveNEW", "TEST6LLRNEW", "TEST6NaiveMultVoteNEW",...
    "TEST6LLRMultVoteNEW", "TEST6LLRVoteRangeNEW"];
modelsall = [models1; models2];
% modelsall = [models1];
savename = ["Figs/MLP7", "Figs/MLPInputReference7"];
titlename = ["MLP Decoders", "Input Reference MLP Decoders"];

seed = 1; % seeding the random number generation for recontruction
rng(seed);
matrix = readmatrix("newMatrix.txt", "Delimiter", " ");
columns = readmatrix("columns.txt", "Delimiter", " ");
columns = columns + 1;
info_loc = columns(1:end/2);
parity_loc = columns(end/2+1:end);
H = matrix;
H2 = H(:, columns);

[temp, order] = sort(columns);
spH2 = sparse(H2);

%%%%%%% make these either 1 or 0, 1 is use, 0 is do not use
use_keras = 1;
%%%%%%%
n_blocks = 1*10^4;
%%%%%%%
SNR = 0:0.5:10;

decoderLDPC = comm.LDPCDecoder('ParityCheckMatrix',spH2);
decoderLDPCC = parallel.pool.Constant(decoderLDPC);
encoder = comm.LDPCEncoder('ParityCheckMatrix',spH2);
encoderC = parallel.pool.Constant(encoder);
demodulator = comm.BPSKDemodulator;
modulator = comm.BPSKModulator;

messages = parallel.pool.Constant(randi([0 1], n_blocks, 100));

if isfile("results/s.mat")
    disp("Found Loadable File")
    decoder = readmatrix("results/decoder.txt");
    hard = readmatrix("results/hard.txt");
    s = load("results/s.mat", "s");
    s = s.s;
else
    %%% BP Decoder and Uncoded
    disp("BP and Uncoded")
    decoder = zeros(size(SNR));
    hard = zeros(size(SNR));
    s = rng;
    parfor i = 1:numel(SNR)
        disp("SNR: " + SNR(i))
        errors_decoder = 0;
        errors_hard = 0;
        tic
        for j = 1:n_blocks
            m = messages.Value(j, :);
            c = GetCodeword(encoderC.Value, info_loc, parity_loc, m);
            c_mod = 1 - 2*c;
            noise = GetNoise(size(c_mod), SNR(i));
            x = c_mod + noise;
            xhat = real(demodulator(x))';
            decoding = x';
            decoded_bp = decoderLDPCC.Value(decoding([info_loc parity_loc])');
            testdecoder_check = xor(decoded_bp', m);

            hard_decoding = xhat;
            testhard_check = xor(hard_decoding(info_loc), m);

            errors_decoder = errors_decoder + sum(testdecoder_check);
            errors_hard = errors_hard + sum(testhard_check);
        end
        decoder(i) = errors_decoder;
        hard(i) = errors_hard;
        toc
    end

    decoder = (decoder/(n_blocks*200))';
    hard = (hard/(n_blocks*200))';
end

for u = 1:size(modelsall,1)
    models = modelsall(u,:)
    labels = [];
    nets = cell(numel(models),1);
    keras = zeros(numel(SNR), numel(nets));
    
    for k = 1:numel(models)
        disp("Testing Model: " + models(k))
        disp("Model " + k + "/" + numel(models))
        file = file_prefix + models(k) + file_suffix;
        %         nets{i} = importKerasNetwork(file);
        net = importKerasNetwork(file);
        rng(s) % ensure the RNG is the same for each SNR 
        parfor i = 1:numel(SNR)
            disp("SNR: " + SNR(i))
            %             errors_keras = zeros(1, numel(nets));
            tic
            for j = 1:n_blocks
                m = messages.Value(j, :);
                c = GetCodeword(encoderC.Value, info_loc, parity_loc, m);
                c_mod = 1 - 2*c;
                noise = GetNoise(size(c_mod), SNR(i));
                x = c_mod + noise;
                received = x;
                naiveQ = Schemes.interpret_demod_bpsk(x',0);
                xhat = real(demodulator(x))';
                votes = GetVotes(H, xhat);
                x = GetLLR(x, SNR(i));
                variance = (1/2)*10^(-SNR(i)/10);
                % getting relevant dataset
                if contains(models(k), string(InputTypes.LLRVoteRange))
                    %note I made the order now (Naive,LLR,Votes) for all novel input schemes in Schemes.m
                    flipped = Schemes.processFlipFromVote(received, x, votes')';
                    arr = [flipped, variance];
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLRVoteRange)];
                    end
                    
                elseif contains(models(k), string(InputTypes.LLRMultVoteMultNaive))
                    %Also: Am I understanding 'Schemes.interpret_demod_bpsk(...)'
                    %correctly?
                    %My idea:
                    LLRMultVoteMultNaive = Schemes.processLLRMultVoteMultNaive(naiveQ',x,votes'); %Schemes.interpret_demod_bpsk(x_thresh,0)
                    arr = [LLRMultVoteMultNaive; variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLRMultVoteMultNaive)];
                    end
                elseif contains(models(k), string(InputTypes.LLRMultVote))
                    %My idea:
                    LLRMultVote = Schemes.processLLRMultVote(x',votes);
                    arr = [LLRMultVote, variance];
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.LLRMultVote)];
                    end
                elseif contains(models(k), string(InputTypes.NaiveMultVote))
                    %My idea:
                    NaiveMultVote = Schemes.processNaiveMultVote(xhat,votes);
                    arr = [NaiveMultVote, variance];
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
                    NaiveVote = [Schemes.interpret_demod_bpsk(received,0), votes];
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
                    arr = [xhat'; variance]';
                    if i == 1 && j == 1
                        labels = [labels, string(InputTypes.Naive)];
                    end
                end
                if contains(models(k), "RNN")
                    arr = arr';
                end
                decoded_netork = predict(net, arr);
                testkeras_round = round(decoded_netork);
                testkeras_check = xor(testkeras_round, m);
                
                %                 errors_keras(k) = errors_keras(k) + sum(testkeras_check);
                keras(i, k) = keras(i, k) + sum(testkeras_check);
            end
            
            %             keras(i, k) = errors_keras;
            
            toc
        end
    end
    
    keras = (keras/(n_blocks*200));
    
    SNR_flip = SNR';
    
    newArr = [SNR_flip, decoder, hard];
    true_labels = ["BP Decoder","Uncoded LDPC"];
    true_labels = [true_labels, ...
        "Naive", "LLR", "NaiveMultVote", "LLRMultVote", "LLRVoteRange"];
    newArr = [newArr, keras];
    
    tableVars = ["SNR", true_labels];
    vars = cellstr(tableVars);
    Decoders = array2table(newArr, 'VariableNames', vars)
    toc
    cmap = colormap(turbo(numel(true_labels)));
    %Optional Plotting
    figure(u)
    plt = semilogy(SNR_flip, newArr(:, 2:end),'LineWidth', 2)
    markers = {'+' ; 'o' ; '*' ; 'd'; 's'; 'x'; 'h'; 'p'};
    [plt(:).Marker] = markers{:};
    for l = 1:size(cmap,1)
        [plt(l).Color] = cmap(l,:);
    end
    set(gca,'FontSize',14)
    x0=623;
    y0=308;
    width=766;
    height=634;
    set(gcf,'position',[x0,y0,width,height])
    grid on
    legend(true_labels);
    xlabel("SNR (dB)");
    ylabel("BER");
    % title("BER vs SNR MLP Decoders")
    title("BER vs SNR " + titlename(u))
    saveas(gcf, savename(u) + ".pdf")
    saveas(gcf, savename(u) + ".fig")
end

