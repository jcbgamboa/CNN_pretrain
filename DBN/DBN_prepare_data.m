function [train_data, trainL, test_data, testL] = DBN_prepare_data( ...
                                    train_data, trainL, test_data, testL)
    % I expect the data to have already been "prepared" for the CNN (i.e.,
    % that CNN_prepare_data() was run already before).

    % Instead of 28 x 28 x N, we want the data to be N x 784. First step is
    % to transform it into 784 x N. Then we just transpose the result.
    train_data = reshape(train_data, 784, 60000)';
    test_data  = reshape(test_data, 784, 10000)';
    
    % No need to process the labels (i.e., trainL and testL) because they
    % were already processed by CNN_prepare_data().
end

