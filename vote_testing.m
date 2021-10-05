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
rcvd = [-7 -4 2 13 3 9 -1];

a = decode_demod_bpsk(rcvd);

function out = decode_demod_bpsk(rcvd)
    out = zeros(size(rcvd));
    for bit = 1:size(rcvd,2)
        if rcvd(bit) <= 0
            out(bit) = 1;
        else
            out(bit) = 0;
        end
    end
end