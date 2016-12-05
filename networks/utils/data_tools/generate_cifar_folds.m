function generate_cifar_folds(...
                input_folder, folds_folder, how_many_folds)

    % Loads data
    if (how_many_folds > 1)
        [data_x, data_y] = load_cifar(input_folder);
        indices = crossvalind('Kfold', size(data_x, 1), how_many_folds);
    else
        [train_data, trainL, test_data, testL] = ...
                load_original(input_folder, add_noise);
    end
    
    % Apparently MATLAB doesn't like this call anymore. Damn it...
    % This reinitializes the seed of the rand() function
    rand('state', 0);

    % Generates the cross validation data
    for i = 1:how_many_folds
        if (how_many_folds > 1)
            [train_data, trainL, test_data, testL] = ...
                get_crossvalidation_data_cifar(data_x, data_y, ...
                                                        indices, i);
        end
        
        % Transform it into images
        train_data = reshape(train_data', 32, 32, 3, 50000);
        test_data = reshape(test_data', 32, 32, 3, 10000);
        
        % Outputs the cross validation data into a file
        % (This is so that the other python scripts can read them)
        output_fold_to_file(folds_folder, i, ...
                    train_data, trainL, test_data, testL);
    end


end

function output_fold_to_file(folds_folder, i, ...
                    train_data, trainL, test_data, testL)
    if (~exist(folds_folder, 'dir'))
        mkdir(folds_folder);
    end
    
    out_filename = strcat(folds_folder, '/fold_', num2str(i), '.mat');
    save(out_filename, 'train_data', 'trainL', 'test_data', 'testL');
end

function [data_x, data_y] = load_cifar(input_folder)
    [train_data, trainL, test_data, testL] = load_cifar_original(input_folder);
    
    data_x = train_data;
    data_x(50001:60000, :) = test_data;
    
    data_y = trainL;
    data_y(50001:60000) = testL;
end

function [train_data, trainL, test_data, testL] = load_cifar_original(input_folder)
    train_data = zeros([50000, 3072], 'uint8');
    trainL = zeros([50000, 1], 'uint8');
    test_data = zeros([10000, 3072], 'uint8');
    testL = zeros([10000, 1], 'uint8');
    
    for i = 1:5
        batch_filename = strcat(input_folder, '/data_batch_', num2str(i), '.mat');
        load(batch_filename);
        train_data(10000*(i-1)+1:10000*i,:) = data;
        trainL(10000*(i-1)+1:10000*i) = labels;
        clear data labels
    end
    
    test_filename = strcat(input_folder, '/test_batch.mat');
    load(test_filename);
    test_data = data;
    testL = labels;
end

function [train_data, trainL, test_data, testL] = ...
                get_crossvalidation_data_cifar(data_x, data_y, ...
                                                indices, curr_fold)
    % Divides the data into `how_many_folds` chunks, and returns one of
    % them as the `test_data`/`test_labels` and the rest as
    % `train_data`/`train_labels`.
    
    % TODO: Obviously this could have been done much more efficiently with
    % some matrix black magic; but I don't have the time now (and whatever
    % gain in efficiency I get from it would not be worth the time and
    % headache lost in making the ritual anyway)...

    % Allocate space
    test_data  = zeros(histc(indices, curr_fold), 3072, 'uint8')';
    train_data = zeros(size(indices, 2) - size(test_data, 1), 3072, 'uint8')';
    
    testL  = zeros(size(test_data, 1), 1, 'uint8');
    trainL = zeros(size(train_data, 1), 1, 'uint8');
    
    curr_test_index = 1;
    curr_train_index = 1;
    for i = 1:size(indices, 1)
        if (indices(i) == curr_fold)
            test_data(:, curr_test_index) = data_x(i, :);
            testL(curr_test_index) = data_y(i);
            curr_test_index = curr_test_index + 1;
        else
            train_data(:, curr_train_index) = data_x(i, :);
            trainL(curr_train_index) = data_y(i);
            curr_train_index = curr_train_index + 1;
        end
    end
    
    test_data = test_data';
    train_data = train_data';
end
