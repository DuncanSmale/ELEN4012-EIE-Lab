function votes = GetVotes(H,codeword)
% H:        parity check matrix
% codeword: received codeword
% votes:    votes for each node

[rownum,colnum]=size(H);
votes = zeros(1,colnum); %this is the final output variable
parity_store = zeros(1,rownum);

%% First we collect the parity checks - pass (0) or fail (1)?
for rowindex = 1:rownum
    parity_tracker = 0;%reset every row
    for colindex = 1:colnum % we are in a specific row, going through columns
        %z = H(rowindex,colindex) prints along column and then row in a Z pattern
        if H(rowindex,colindex) == 1
            %do the corresponding message bits have an even parity?
            if codeword(colindex) == 1
                %disp("One")
                parity_tracker = xor(parity_tracker,codeword(colindex));
            else
                %disp("Zero")
            end
        end
    end
    %parity_tracker % 1 = fail, 0 = pass
    parity_store(rowindex) = parity_tracker;
end
%% Second we allocate which bits have caused those parity checks to fail (i.e. the votes)
for rowindex = 1:rownum
    for colindex = 1:colnum
        if H(rowindex,colindex) == 1
            votes(colindex) = votes(colindex) + parity_store(rowindex);
        end
    end
end
%votes % the desired output

