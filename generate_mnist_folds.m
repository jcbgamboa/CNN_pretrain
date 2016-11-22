function generate_mnist_folds(...
                input_file_name, folds_folder, how_many_folds)

    % Loads data
    add_noise = 0;
    if (how_many_folds > 1)
        [data_x, data_y] = load_data(input_file_name, add_noise);
        indices = crossvalind('Kfold', size(data_x, 4), how_many_folds);
    else
        [train_data, trainL, test_data, testL] = ...
                load_data_mnist_original(input_file_name, add_noise);
    end
    
    % Apparently MATLAB doesn't like this call anymore. Damn it...
    % This reinitializes the seed of the rand() function
    rand('state', 0);

    % Generates the cross validation data
    for i = 1:how_many_folds
        if (how_many_folds > 1)
            [train_data, trainL, test_data, testL] = ...
                get_crossvalidation_data_mnist(data_x, data_y, ...
                                                        indices, i);
        end
        
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
