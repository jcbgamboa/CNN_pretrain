function cnn = CNN_transfer_weights_CDBN(cnn, model, layers_info)
% Transfers the weights from the CDBN into the CNN

    % First, we transfer the biases
    for i = 1:9
        cnn.layers{2}.k{1}{i} = model{1}.W(:,:,1,i);
        cnn.layers{2}.b{i}    = model{1}.h_bias(i);
    end
    
    for i = 1:16
        for j = 1:9
            cnn.layers{4}.k{j}{i} = model{2}.W(:,:,j,i);
        end
        cnn.layers{4}.b{i} = model{2}.h_bias(i);
    end
    
    % TODO: Find a way to make the code find out by itself what are the
    % layers to which it has to transfer weights. For now, everything is
    % hardcoded. Probably I should use a `layers_info` variable.
end

