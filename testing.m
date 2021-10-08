clc
matrix = readmatrix("newMatrix.txt", "Delimiter", " ");
columns = readmatrix("columns.txt", "Delimiter", " ");
columns = columns + 1;
info_loc = columns(1:end/2);
parity_loc = columns(end/2+1:end);
H = matrix;
H2 = H(:, columns);
m = randi([0 1], 1, 100);

[temp, order] = sort([info_loc parity_loc]);
spH2 = sparse(H2);
encoder = comm.LDPCEncoder('ParityCheckMatrix',spH2);
decoder = comm.LDPCDecoder('ParityCheckMatrix',spH2);
inter = step(encoder, m');

codeword = inter(order);
[codeword(info_loc) m'];

modulated = 1 - 2*codeword;
decoded = decoder(modulated([info_loc parity_loc]));
check = decoded == m';
correct = all(check(:))

% Test whether codeword is right. No error should occur
if max(abs(mod(H*codeword,2)))>0
    error('Parity-check equations violated');
end
if max(abs(codeword(info_loc)- m'))>0
    error('Information bits mismatched');
end


% load H.mat H_rev
% H = H_rev;
% H = H2;
% rows=size(H,1);
% cols=size(H,2);
% 
% O=H*H';   
% for i=1:rows
%     O(i,i)=0;
% end
% for i=1:rows
% girth(i)=max(O(i,:));
% end
% girth4=max(girth);
% if girth4<2 
%   fprintf('No girth 4')
% else
%    fprintf('The H matrix has girth 4')  % Provde the test result.
% end    
% % Display the matrice H and O
% % If H matrix has no gith4, the O matrix in Fig.2 has no entry value to
% % larger than 1.
% figure(1)
% mesh(H)
% figure(2)
% mesh(O)