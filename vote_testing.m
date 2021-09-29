clc;
%note: in this toy example, the only valid codewords are '0000000' and
%'1111111'
H = [1 1 0 0 0 0 0;
    0 1 1 0 0 0 0;
    0 1 1 1 1 0 0;
    0 0 0 1 1 0 0;
    0 0 0 0 1 1 0;
    0 0 0 0 1 0 1];
% H = [0 1 0 1 1 0 0 1;
%     1 1 1 0 0 1 0 0;
%     0 0 1 0 0 1 1 1;
%     1 0 0 1 1 0 1 0];
[rownum,colnum]=size(H);

%msg = [0 0 0 0 0 0 0];
rcvd_msg = [0 1 0 0 1 0 0];
%rcvd_msg = [0 1 0 1 1 0 0 0];

GetVotes(H,rcvd_msg)
% votes = zeros(1,colnum); %this is the final output
% parity_store = zeros(1,rownum);
% 
% %% First we collect the parity checks - pass (0) or fail (1)?
% for rowindex = 1:rownum
%     parity_tracker = 0;%reset every row
%     for colindex = 1:colnum % we are in a specific row, going through columns
%         %z = H(rowindex,colindex) prints along column and then row in a Z pattern
%         if H(rowindex,colindex) == 1
%             %do the corresponding message bits have an even parity?
%             if rcvd_msg(colindex) == 1
%                 %disp("One")
%                 parity_tracker = xor(parity_tracker,rcvd_msg(colindex));
%             else
%                 %disp("Zero")
%             end
%         end
%     end
%     %parity_tracker % 1 = fail, 0 = pass
%     parity_store(rowindex) = parity_tracker;
% end
% 
% %parity_store
% %% Second we allocate which bits have caused those parity checks to fail (i.e. the votes)
% for rowindex = 1:rownum
%     for colindex = 1:colnum
%         if H(rowindex,colindex) == 1
%             votes(colindex) = votes(colindex) + parity_store(rowindex);
%         end
%     end
% end
% votes % the desired output

