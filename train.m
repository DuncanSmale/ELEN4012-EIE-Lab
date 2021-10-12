clc
type = InputTypes.NaiveMultVote;
samples = '10K';
SNR = '_0_3SNR100';
prefix = './' + string(type) + '/' + samples + SNR;
data = readmatrix(prefix + 'data' + 'TRAIN' + string(type) + '.txt', 'Delimiter', ',');
messages = readmatrix(prefix + 'messages' + 'TRAIN' + string(type) + '.txt', 'Delimiter', ',');
Xtest = readmatrix(prefix + 'data' + 'TEST' + string(type) + '.txt', 'Delimiter', ',');
Ytest = readmatrix(prefix + 'messages' + 'TEST' + string(type) + '.txt', 'Delimiter', ',');

miniBatchSize = 16;
opts = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1*10^-5, ...
    'MaxEpochs',1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{Xtest,Ytest});
N = 200;
NUM_HIDDEN = 20*N;
dropout = 0.4;

layers = [
    featureInputLayer(N+1)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    layerNormalizationLayer
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(100)
    sigmoidLayer
    regressionLayer];

net = trainNetwork(data, messages, layers, opts);
name = string(type) + '.mat';
save(name, 'net')
% YPred = round(predict(net, Xtest));
% errors = sum(sum(xor(Ytest,YPred)));