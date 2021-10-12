clc
data = readmatrix('./LLR/10K_0_3SNR100dataTRAINLLR.txt', 'Delimiter', ',');
messages = readmatrix('./LLR/10K_0_3SNR100messagesTRAINLLR.txt', 'Delimiter', ',');
Xtest = readmatrix('./LLR/10K_0_3SNR100dataTESTLLR.txt', 'Delimiter', ',');
Ytest = readmatrix('./LLR/10K_0_3SNR100messagesTESTLLR.txt', 'Delimiter', ',');

miniBatchSize = 16;
opts = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1*10^-5, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{Xtest,Ytest});
N = 200;
NUM_HIDDEN = 10*N;
dropout = 0.4;

layers = [
    featureInputLayer(N+1)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(NUM_HIDDEN)
    tanhLayer
    dropoutLayer(dropout)
    fullyConnectedLayer(100)
    sigmoidLayer
    regressionLayer];

net = trainNetwork(features, response, layers, opts);
save net
% YPred = round(predict(net, Xtest));
% errors = sum(sum(xor(Ytest,YPred)));