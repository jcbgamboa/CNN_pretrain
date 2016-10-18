function [train_data, train_labels, test_data, test_labels] = ...
            get_crossvalidation_data_mnist(data_x, data_y, ...
                                    indices, curr_fold)
    % Divides the data into `how_many_folds` chunks, and returns one of
    % them as the `test_data`/`test_labels` and the rest as
    % `train_data`/`train_labels`.
    
    % Allocate space
    test_data  = zeros(28, 28, 1, histc(indices, curr_fold));
    train_data = zeros(28, 28, 1, size(indices, 2) - size(test_data, 2));
    
    test_labels  = zeros(size(test_data, 4), 1);
    train_labels = zeros(size(train_data, 4), 1);

    curr_test_index = 1;
    curr_train_index = 1;
    for i = 1:size(indices, 1)
        if (indices(i) == curr_fold)
            test_data(:, :, 1, curr_test_index) = data_x(:, :, 1, i);
            test_labels(curr_test_index) = data_y(i);
            curr_test_index = curr_test_index + 1;
        else
            train_data(:, :, 1, curr_train_index) = data_x(:, :, 1, i);
            train_labels(curr_train_index) = data_y(i);
            curr_train_index = curr_train_index + 1;
        end
    end
    
    % TODO: Obviously this could have been done much more efficiently with
    % some matrix black magic; but I don't have the time now (and whatever
    % gain in efficiency I get from it would not be worth the time and
    % headache lost in making the ritual anyway)...
end

