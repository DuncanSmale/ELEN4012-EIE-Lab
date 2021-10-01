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
a = [1 2 3 4;11 22 33 44; 1 2 3 4; 11 22 33 44]'
b = [5 6; 55 66]'
y = [b b]
c = [a;y]
%y = zeros(2*size(a))

%c = [a;b]
