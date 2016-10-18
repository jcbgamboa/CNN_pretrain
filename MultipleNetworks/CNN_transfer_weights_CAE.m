function cnn = CNN_transfer_weights_CAE(cnn, file_name)
    load(file_name);
    W1 = permute(caes_W1, [3, 4, 2, 1]);
    W2 = permute(caes_W2, [3, 4, 2, 1]);

    for i = 1:9
        cnn.layers{2}.k{1}{i} = W1(:, :, 1, i);
        cnn.layers{2}.b{i}    = caes_b1(i);
    end
    
    for i = 16
        for j = 1:9
            cnn.layers{4}.k{j}{i} = W2(:,:, j, i);
        end
        cnn.layers{4}.b{i} = caes_b2(i);
    end
end
