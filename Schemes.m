%This class holds novel input schemes and other useful methods
classdef Schemes
    methods(Static)
        function datasetOut = processNaiveMultVote(NaiveValues,VoteValues)
            max_votes = max(max(VoteValues));
            datasetOut = (max_votes-VoteValues+1) .* NaiveValues;
        end

        function datasetOut = processLLRMultVoteMultNaive(NaiveValues,LLRValues,VoteValues)%dataset,tempvotes,x_naive)
            max_votes = max(max(VoteValues));
            datasetOut = LLRValues .* (max_votes-VoteValues+1) .* NaiveValues;
        end

        function datasetOut = processLLRMultVote(LLRValues,VoteValues)
            max_votes = max(max(VoteValues));
            datasetOut = LLRValues .* (max_votes-VoteValues+1);
        end

        function datasetOut = processFlipFromVote(NaiveValues, LLRValues, VoteValues)
            threshold = 1;
            datasetOut = LLRValues;
            ind_LLR =  find(abs(NaiveValues)<threshold);
            ind_votes = find(VoteValues>2);
            lia = ismember(ind_LLR, ind_votes);
            ind = ind_LLR .* lia;
            vals = ind(ind~=0);
            datasetOut(vals) = -datasetOut(vals);
        end

        %interpretAndDecode must be either true (1) or false (0) only.
        function out = interpret_demod_bpsk(rcvd, interpretAndDecode)
            out = zeros(size(rcvd));
            %interprets and Demods - i.e. converts -9,872 to -1 to 1
            if interpretAndDecode == true
                for bit = 1:size(rcvd,2)
                    if rcvd(bit) < 0
                        out(bit) = 1;
                    else
                        out(bit) = 0;
                    end
                end
            %Only interprets - i.e. converts -9.872 to -1
            elseif interpretAndDecode == false
                for bit = 1:size(rcvd,2)
                    if rcvd(bit) < 0
                        out(bit) = -1;
                    else
                        out(bit) = 1;
                    end
                end
            else 
                disp("interpretAndDecode must be either true (1) or false (0) only.");
            end
        end
        
    end
end

% function out = FlipFromVote(received, votes, LLR)
% threshold = 0.1;
% out = LLR;
% ind_LLR =  find(abs(received)<threshold);
% ind_votes = find(votes>2);
% lia = ismember(ind_LLR, ind_votes);
% ind = ind_LLR .* lia;
% vals = ind(ind~=0);
% out(vals) = -out(vals);
% end