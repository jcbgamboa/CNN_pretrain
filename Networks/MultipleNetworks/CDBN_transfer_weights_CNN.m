function model = CDBN_transfer_weights_CNN(cnn, model, layers_info)
% Transfers the weights from the CNN into the CDBN

    % First, we transfer the biases
    for i = 1:9
        model{1}.W(:,:,1,i) = cnn.layers{2}.k{1}{i};
        model{1}.h_bias(i)  = cnn.layers{2}.b{i};
    end

    for i = 1:16
        for j = 1:9
            model{2}.W(:,:,j,i) = cnn.layers{4}.k{j}{i};
        end
        model{2}.h_bias(i) = cnn.layers{4}.b{i};
    end
end

