function [cnn, errors] = run_networks_mnist(...
            input_file_name, mode, how_many_folds, this_opts)
    % Runs the networks in the mnist dataset
    %

    % `mode` can have the following values
    % 0: Random weight initialization
    % 1: Pretrain with CDBN and use its weights to initialize the CNN
    % 2: Pretrain with DBN and use its weights to initialize the CNN
    % 3: Pretrain with CAE (using python) and use its weights to initialize
    %       the CNN.
    % 4: Intercalate training with CDBN and CNN, using the weights of one of
    %       them to initialize the other.

    % `how_many_folds` indicates the number of partitions of the data in
    % the crossvalidation. E.g., `how_many_folds = 7` means the testing
    % will be a 7-fold crossvalidation.
    
    % If `mode` is 4, we want to know how many times the entire procedure
    % should be repeated. This is done with `intercalations`.

    % Loads data
    add_noise = 0;
    [data_x, data_y] = load_data(input_file_name, add_noise);

    % Apparently MATLAB doesn't like this call anymore. Damn it...
    rand('state', 0);
    
    errors = [];
    indices = crossvalind('Kfold', size(data_x, 4), how_many_folds);
    for i = 1:how_many_folds
        [train_data, trainL, test_data, testL] = ...
            get_crossvalidation_data_mnist(data_x, data_y, ...
                                    indices, i);

        % Initializes the CNN
        [CNN_train_data, CNN_trainL, CNN_test_data, CNN_testL] = ...
                CNN_prepare_data(train_data, trainL, test_data, testL);

        if (exist('this_opts', 'var') && isfield(this_opts, 'cnn_training_size'))
                CNN_train_data = CNN_train_data(:, :, 1:this_opts.cnn_training_size);
                CNN_trainL = CNN_trainL(:, 1:this_opts.cnn_training_size);
        end

        opts.alpha = 1;
        opts.batchsize = 50; %1;
        opts.numepochs = 10;
        opts.dims = 2;
        cnn.layers = {
            struct('type', 'i') %input layer
            struct('type', 'c', 'outputmaps', 9,  'kernelsize', [7,7]) %convolution layer
            struct('type', 's', 'scale', [2,2]) %sub sampling layer
            struct('type', 'c', 'outputmaps', 16, 'kernelsize', [6,6]) %convolution layer
            struct('type', 's', 'scale', [2,2]) %subsampling layer
        };
        cnn = cnnsetup(cnn, CNN_train_data, CNN_trainL, opts);


        % Pretrain with CDBN
        if (mode == 1)
            % Initializes the CDBN
            filter_sizes{1} = [7 7];
            filter_sizes{2} = [6 6];
            pooling_sizes{1} = [2 2];
            pooling_sizes{2} = [2 2];
            n_epochs = [10 10];
            input_types{1} = 'Binary';
            input_types{2} = 'Binary';
            cdbn_layers = CDBN_init(train_data, filter_sizes, ...
                                pooling_sizes, n_epochs, input_types);
            
            % Runs the CDBN
            [cdbn_model, cdbn_layers] = run_CDBN(cdbn_layers, ...
                                train_data, test_data);
            cnn = CNN_transfer_weights_CDBN(cnn, cdbn_model);
        elseif (mode == 2)
            % THIS IS NOT WORKING! LEAVE THIS FOR NOW...

            %[DBN_train_data, DBN_trainL, DBN_test_data, DBN_testL] = ...
            %    CNN_prepare_data(train_data, trainL, test_data, testL);
            %[DBN_train_data, DBN_trainL, DBN_test_data, DBN_testL] = ...
            %    DBN_prepare_data(DBN_train_data, DBN_trainL, ...
            %                        DBN_test_data, DBN_testL);
            %[W1, vbias1, hbias1, W2, vbias2, hbias2, pars] = sparse_dbn(...
            %    28, [4356, 576], 0.03, 'simple', 100, train_data');
            %weights{1} = W1;
            %weights{2} = W2;
            %cnn = CNN_transfer_weights_DBN(cnn, weights);
        elseif (mode == 3)
            % Save data into .mat file
            CAES_train_data = single(permute(train_data, [4, 3, 1, 2]));
            save('caes_train_data.mat', 'CAES_train_data');
            clear CAES_train_data;
            
            % Call python code (using system calls, maybe?)
            system('./caes.py');
            
            % Get the resulting .mat file (with Weights)
            % Insert them into the CNN -- actually done later
            cnn = CNN_transfer_weights_CAE(cnn, ...
                                'convolutional_autoencoder/caes_out.mat');
        elseif (mode == 4)
            % Initializes the CDBN
            filter_sizes{1} = [7 7];
            filter_sizes{2} = [6 6];
            pooling_sizes{1} = [2 2];
            pooling_sizes{2} = [2 2];
            n_epochs = [2 2];
            input_types{1} = 'Binary';
            input_types{2} = 'Binary';
            cdbn_layers = CDBN_init(train_data, filter_sizes, ...
                                    pooling_sizes, n_epochs, input_types);

            % Runs the CDBN for the first time
            [cdbn_model, cdbn_layers] = run_CDBN(cdbn_layers, ...
                            train_data, test_data);
            cnn = CNN_transfer_weights_CDBN(cnn, cdbn_model);
            
            for j = 1:this_opts.intercalations-1
                cnn = cnntrain(cnn, CNN_train_data, CNN_trainL, opts);
                cdbn_model = CDBN_transfer_weights_CNN(cnn, cdbn_model);
                
                [cdbn_model, cdbn_layers] = run_CDBN(cdbn_layers, ...
                                    train_data, test_data, cdbn_model);
                cnn = CNN_transfer_weights_CDBN(cnn, cdbn_model);
            end
        end

        
        cnn = cnntrain(cnn, CNN_train_data, CNN_trainL, opts); 

        % We take the test set as only one huge test set
        opts.batchsize = 10000;
        [er, bad] = cnntest(cnn, CNN_test_data, CNN_testL, opts);
        
        % Stores the number of errors in this variable
        n_errors = size(bad);
        errors(i) = n_errors(2);
        %plot mean squared error
        %figure; plot(cnn.rL);
    end

end
