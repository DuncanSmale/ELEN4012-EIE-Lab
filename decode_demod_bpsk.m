%decodeAndDecode must be either true (1) or false (0) only.
function out = decode_demod_bpsk(rcvd, decodeAndDecode)
    out = zeros(size(rcvd));
    %Decodes and Demods - i.e. converts -9,872 to -1 to 1
    if decodeAndDecode == true
        for bit = 1:size(rcvd,2)
            if rcvd(bit) <= 0
                out(bit) = 1;
            else
                out(bit) = 0;
            end
        end
    %Only decodes - i.e. converts -9.872 to -1
    elseif decodeAndDecode == false
        for bit = 1:size(rcvd,2)
            if rcvd(bit) <= 0
                out(bit) = -1;
            else
                out(bit) = 1;
            end
        end
    else 
        disp("decodeAndDecode must be either true (1) or false (0) only.");
    end
    

end