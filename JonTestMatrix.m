clc
% % this for loop helped generate a non singular A matrix 
% for i = 1:1000
%     cols = randi([1 10], 1, 2);
%     h = H(:,cols(1));
%     H(:,cols(1)) = H(:,cols(2));
%     H(:,cols(2)) = h;
%     A = gf(H(:, 1:end/2));
%     if det(A) ~= 0
%         break
%     end
% end

% save('test_matrix.mat', 'H')

load('test_matrix.mat', 'H');

B_gf = gf(H(:, 1:end/2));
A_gf = gf(H(:, end/2 + 1:end));

A_inv = inv(A_gf);

% generate codeword
m = randi([0 1], 1, 5);
m_gf = gf(m);

% multiply A_inv B and m
% check = A_inv * B_gf * m_gf';
% these two are used to verify that the encoding of built in encoder and
% our mathematical method is correct
% creating code word
% c = [m_gf'; check];

[row, col] = find(H);
I = [row col];
index = sparse(I(:,1),I(:,2),1);
encoder = comm.LDPCEncoder('ParityCheckMatrix',index);
decoder = comm.LDPCDecoder('ParityCheckMatrix',index);
