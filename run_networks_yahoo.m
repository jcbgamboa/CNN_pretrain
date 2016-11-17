function run_networks_yahoo(dataset_folder, mode)
    % Runs the networks in the Yahoo! dataset

    % `mode` can have the following values
    % 0: Random weight initialization
    % 1: Pretrain with CDBN and use its weights to initialize the CNN

    files = getAllFiles(dataset_folder);
    how_many_files = 6;
    files = files(1:how_many_files);

    % Initialize a variable to store the models (needed for `mode = 1`)
    
    cdbn_models = cell([how_many_files, 4]);
    for i = 1:length(files)
        [train_wins, train_wins_labels, test_wins, test_wins_labels] ...
                                    = read_yahoo_file(cell2mat(files(i)));

        if (mode == 1)
            % Initializes the CDBN
            filter_sizes{1} = [5 1];
            filter_sizes{2} = [5 1];
            pooling_sizes{1} = [1 1];
            pooling_sizes{2} = [1 1];
            n_epochs = [50 50];
            input_types{1} = 'Gaussian';
            input_types{2} = 'Gaussian';
            cdbn_layers = CDBN_init(train_wins, filter_sizes, ...
                                    pooling_sizes, n_epochs, input_types);

            % The Lasagne code is using 20 filters in the first layer, and
            % 10 in the second layer. Hardcode these numbers here for now.
            cdbn_layers{1}.n_map_v = 1;
            cdbn_layers{1}.n_map_h = 20;
            cdbn_layers{2}.n_map_v = 20;
            cdbn_layers{2}.n_map_h = 10;

            [cdbn_model, cdbn_layers] = run_CDBN(...
                                    cdbn_layers, train_wins, train_wins);

            % Stores the model in a variable -- so that later we can save
            % it along with all others into a file that will be read by the
            % Lasagne CNN implementation
            cdbn_models{i,1} = cdbn_model{1,1}.W;
            cdbn_models{i,2} = cdbn_model{1,1}.h_bias;
            cdbn_models{i,3} = cdbn_model{1,2}.W;
            cdbn_models{i,4} = cdbn_model{1,2}.h_bias;
        end
    end
    
    if (mode == 1 || mode == 3)
        % Save data into .mat file
        save('cdbn_models.mat', 'cdbn_models');
    end
    
    % Runs Lasagne CNNs (in plural because it will run on all files)
    system('./cnn_lasagne.py');
end

function [train_wins, train_wins_labels, ...
                test_wins, test_wins_labels] = read_yahoo_file(file_name)
    % Reads a csv file from the Yahoo! dataset
    file_name = './yahoo_dataset/A1Benchmark/real_4.csv';
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

    % NORMALIZE THE TRAINING DATA
    % (because the CDBN uses binary units that will not output anything
    % outside the range [0, 1])
    % [I am not caring about the test data because it is processed
    % somewhere else, not by Matlab]
    
    % First, we "zero center" the data
    train_wins = train_wins - min(train_wins(:));
    train_wins_labels = train_wins_labels - min(train_wins_labels);
    
    % Then we scale it to be in the range [0, 1]
    train_wins = train_wins ./ max(train_wins(:));
    train_wins_labels = train_wins_labels ./ max(train_wins_labels);
end
