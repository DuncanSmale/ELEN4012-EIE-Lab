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

%msg = [0 0 0 0 0 0 0];
rcvd_msg = [0 1 0 0 1 0 0];
%rcvd_msg = [0 1 0 1 1 0 0 0];

%GetVotes(H,rcvd_msg)
%a = [1 2 3 4;11 22 33 44; 1 2 3 4; 11 22 33 44]'
%b = [5 6; 55 66]'
%y = [b b]
%c = [a;y]
%y = zeros(2*size(a))

%c = [a;b]

%a = [1 2;3 4]
%m = max(max(a))
%b = [0 0.2;0.3 0.4]
%c = a .* 1./(1+b)

% 1 ==> 0
% -1 ==> 1
% rcvd = [-7 -4 2 13 3 9 -1];
% 
% a = decode_demod_bpsk(rcvd);
% 
% function out = decode_demod_bpsk(rcvd)
%     out = zeros(size(rcvd));
%     for bit = 1:size(rcvd,2)
%         if rcvd(bit) <= 0
%             out(bit) = 1;
%         else
%             out(bit) = 0;
%         end
%     end
% end

%% Tanner Graph Diagram
load H.mat H_rev
H = H_rev;
% H = [H;H];
%Make a random MxN adjacency matrix
m = 100;
n = 200;
a = rand(m,n)>.25;
% Expand out to symmetric (M+N)x(M+N) matrix
%This line is a big of a fib - this matrix is not H, but big_a
%Use as pretty picture, not as an actual diagram
big_a = [zeros(m,m), H;
         H', zeros(n,n)];     


g = graph(big_a);
% Plot
h = plot(g);
% Make it pretty
h.XData(1:m) = 1;
h.XData((m+1):end) = 2;
h.YData(1:m) = linspace(0,1,m);
h.YData((m+1):end) = linspace(0,1,n);