function datasetOut = GetVoteDataset(dataset,x,percent_noisy,noisy_index,demodulator,H)
    temp_votes = zeros(size(dataset));
    %demodulate noisy part
    for j = 1:size(x,2)
        x(:, j) = real(demodulator(x(:, j)))';
    end

    if percent_noisy ~= 0
        %Add the nosiy part to dataset
        dataset(:, noisy_index + 1: end) = x;
        %calculate the votes for non-noisy (should be all 0) and noisy parts
        for j = 1:size(dataset,2)
            temp_votes(:,j) = GetVotes(H,dataset(:,j));
        end
        %Replace dataset with only votes 
        datasetOut = temp_votes;
    end
end