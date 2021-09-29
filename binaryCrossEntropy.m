classdef binaryCrossEntropy < nnet.layer.ClassificationLayer
methods
        function layer = binaryCrossEntropy(name) 
            % Set layer name
            if nargin == 1
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'sigmoidLayer'; 
        end
        function Z = predict(layer,X)
            % Forward input data through the layer and output the result
            Z = sigmoid(X);
        end
        % No need to define a backward function, as MATLAB supports automatic differentiation of sigmoid
%         function dLdX = backward(layer, X ,Z,dLdZ,memory)
%             % Backward propagate the derivative of the loss function through 
%             % the layer 
%             dLdX = Z.*(1-Z) .* dLdZ;
%         end

        function forwardLoss(layer, Y, T)
            
        end
    end
end

