function votes = GetVotes(H,codeword)
% H:        parity check matrix
% codeword: received codeword
% votes:    votes for each node
votes = zeros(size(codeword));

% loop over each row and column
% find what each parity check node requires from the matrix
% check if each codeword satisfies the parity check node
% if it doesnt 
[n,m] = size(H);
for i = 1:n
    for j = 1:m
        
    end
end
end

