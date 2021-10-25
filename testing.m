clc
% 
% seed = 1; % seeding the random number generation for recontruction
% rng(seed);
% matrix = readmatrix("newMatrix.txt", "Delimiter", " ");
% columns = readmatrix("columns.txt", "Delimiter", " ");
% columns = columns + 1;
% info_loc = columns(1:end/2);
% parity_loc = columns(end/2+1:end);
% H = matrix;
% H2 = H(:, columns);
% 
% [temp, order] = sort(columns);
% spH2 = sparse(H2);
% 
% %%%%%%% make these either 1 or 0, 1 is use, 0 is do not use
% use_keras = 1;
% %%%%%%%
% n_blocks = 1*10^5;
% %%%%%%%
% SNR = 0:0.5:10;
% 
% decoderLDPC = comm.LDPCDecoder('ParityCheckMatrix',spH2);
% decoderLDPCC = parallel.pool.Constant(decoderLDPC);
% encoder = comm.LDPCEncoder('ParityCheckMatrix',spH2);
% encoderC = parallel.pool.Constant(encoder);
% demodulator = comm.BPSKDemodulator;
% modulator = comm.BPSKModulator;
% 
% messages = parallel.pool.Constant(randi([0 1], n_blocks, 100));
% 
% %%% BP Decoder and Uncoded
% disp("BP and Uncoded")
% decoder = zeros(size(SNR));
% hard = zeros(size(SNR));
% s = rng;
% parfor i = 1:numel(SNR)
%     disp("SNR: " + SNR(i))
%     errors_decoder = 0;
%     errors_hard = 0;
%     tic
%     for j = 1:n_blocks
%         m = messages.Value(j, :);
%         c = GetCodeword(encoderC.Value, info_loc, parity_loc, m);
%         c_mod = 1 - 2*c;
%         noise = GetNoise(size(c_mod), SNR(i));
%         x = c_mod + noise;
%         xhat = real(demodulator(x))';
%         decoding = x';
%         decoded_bp = decoderLDPCC.Value(decoding([info_loc parity_loc])');
%         testdecoder_check = xor(decoded_bp', m);
%         
%         hard_decoding = xhat;
%         testhard_check = xor(hard_decoding(info_loc), m);
%         
%         errors_decoder = errors_decoder + sum(testdecoder_check);
%         errors_hard = errors_hard + sum(testhard_check);
%     end
%     decoder(i) = errors_decoder;
%     hard(i) = errors_hard;
%     toc
% end
% 
% decoder = (decoder/(n_blocks*200))';
% hard = (hard/(n_blocks*200))';
% 
% if ~exist("results", 'dir')
%     disp("Creating results directory")
%     mkdir("results")
% end
% save("results/s.mat", "s");
% writematrix(decoder, "results/decoder.txt")
% writematrix(hard, "results/hard.txt")

savename = ["Figs/MLP8", "Figs/MLPInputReference8"];
titlename = ["MLP Decoders", "Input Reference MLP Decoders"];
filenamesave = ["MLP.txt", "MLPInputReference.txt"];

decoder = readmatrix("results/decoder.txt");
hard = readmatrix("results/hard.txt");
MLP = readmatrix("results/MLP.txt");
MLPInput = readmatrix("results/MLPInputReference.txt");

newArr = [decoder, hard];
true_labels = ["BP Decoder","Uncoded LDPC"];
true_labels = [true_labels, ...
    "Naive", "LLR", "NaiveMultVote", "LLRMultVote", "LLRVoteRange"];
newArr = [newArr, MLPInput];

cmap = colormap(turbo(numel(true_labels)));
cmap = cmap(2:end,:);
%Optional Plotting
u=2;
figure(u)
plt = semilogy(SNR_flip, newArr(:, 2:end),'LineWidth', 2)
markers = {'o' ; '*' ; 'd'; 's'; 'x'; 'h'; 'p'}; %'+' ; 
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
legend(true_labels(2:end));
xlabel("SNR (dB)");
ylabel("BER");
title("BER vs SNR " + titlename(u))
