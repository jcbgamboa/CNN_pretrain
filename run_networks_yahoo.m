function [cnn, errors] = run_networks_yahoo(dataset_folder, mode)
    % Runs the networks in the Yahoo! dataset

    % `mode` can have the following values
    % 0: Random weight initialization
    % 1: Pretrain with CDBN and use its weights to initialize the CNN

    files = getAllFiles(dataset_folder);

    for i = 1:length(files)
        [train_wins, train_wins_labels, test_wins, test_wins_labels] ...
                                    = read_yahoo_file(cell2mat(files(i)));
                                
        opts.alpha = 0.01;
        opts.batchsize = 20;
        opts.numepochs = 100;
        opts.dims = 1;
        opts.regression = 1;
        cnn.layers = {
            struct('type', 'i') %input layer
            struct('type', 'c', 'outputmaps', 20, 'kernelsize', 11) %convolution layer
            struct('type', 's', 'scale', 2) %sub sampling layer
            struct('type', 'c', 'outputmaps', 100, 'kernelsize', 2) %convolution layer
            struct('type', 's', 'scale', 2) %subsampling layer
        };
        cnn = cnnsetup(cnn, train_wins, train_wins_labels, opts);

        if (mode == 1)
            % Initializes the CDBN
            filter_sizes{1} = [11 1];
            filter_sizes{2} = [2 1];
            pooling_sizes{1} = [2 1];
            pooling_sizes{2} = [2 1];
            n_epochs = [2 2];
            input_types{1} = 'Gaussian';
            input_types{2} = 'Gaussian';
            cdbn_layers = CDBN_init(train_wins, filter_sizes, ...
                                    pooling_sizes, n_epochs, input_types);

            [cdbn_model, cdbn_layers] = run_CDBN(...
                                    cdbn_layers, train_wins, train_wins);
            cnn = CNN_transfer_weights_CDBN(cnn, cdbn_model);
        end

        cnn = cnntrain(cnn, train_wins, train_wins_labels, opts); 

        % We need to rewrite the function that tests the cnn. Instead of
        % classification, we now want to know the regression error.
        opts.batchsize = size(test_wins_labels, 2);
        [err, bad] = CNN_regression_test(cnn, test_wins, ...
                                        test_wins_labels, opts);

        %[err, bad] = cnntest(cnn, test_wins, test_wins_labels, opts);
        %bads{i, 1} = bad;
        %bads{i, 2} = find(test_wins_labels(2,:) == 1);
    end

    %plot mean squared error
    %figure; plot(cnn.rL);
end

function [train_wins, train_wins_labels, ...
                test_wins, test_wins_labels] = read_yahoo_file(file_name)
    % Reads a csv file from the Yahoo! dataset
    file_name = './project_code/yahoo_dataset/A1Benchmark/real_4.csv';
    fprintf('Reading file "%s"\n', file_name);
    
    csv_data = csvread(file_name, 1);
    train_data = csv_data(:,2);
    trainL = csv_data(:,3);

    % Gets the input data. We leave the last time-step out: it is going to
    % be the label for the last input window.
    train_wins  = train_data(1:399);
    test_wins   = train_data(401:end-1);

    % Converts the input data into windows.
    train_wins  = createRollingWindow(train_wins, 20)';
    test_wins   = createRollingWindow(test_wins, 20)';

    % Gets the "labels" (actually, since we are doing regression, we want
    % to train the network to predict the next value of the time series).
    train_wins_labels = train_data(21:400)';
    test_wins_labels  = train_data(421:end)';

    %train_winsL = labelmatrix(train_winsL(1:end-19), 2)';
    %test_winsL  = labelmatrix(test_winsL(1:end-19), 2)';

end
