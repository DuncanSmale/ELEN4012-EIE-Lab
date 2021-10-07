function out = FlipFromVote(received, votes, LLR)
threshold = 0.1;
out = LLR;
ind_LLR =  find(abs(received)<threshold);
ind_votes = find(votes>2);
lia = ismember(ind_LLR, ind_votes);
ind = ind_LLR .* lia;
vals = ind(ind~=0);
out(vals) = -out(vals);
end

