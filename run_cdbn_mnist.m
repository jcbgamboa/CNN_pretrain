function run_cdbn_mnist(how_many_folds, input_folder, output_folder, opts)
    % Runs the CDBN for each of the cross-validation folders of the input.

    % I totally plan to abuse this function and run CIFAR-10 with it too
    for i = 1:how_many_folds
        [train_data, trainL, test_data, testL] = ...
                                    load_mnist_data(input_folder, i);

        % I am saving as much memory as I can <o>
        clear trainL test_data testL;

        if(opts.normalize == 1)
            % Unfortunately, we need to make this sometimes (which will
            % multiply by 8 the amount of bytes needed to store the data)
            train_data = double(train_data) / double(255);
        end

        % Initializes the CDBN
        filter_sizes{1} = [7 7];
        filter_sizes{2} = [6 6];
        pooling_sizes{1} = [2 2];
        pooling_sizes{2} = [2 2];
        n_epochs = [1 1];
        input_types{1} = opts.cdbn_unit_type;
        input_types{2} = opts.cdbn_unit_type;
        cdbn_layers = CDBN_init(train_data, filter_sizes, ...
                        pooling_sizes, n_epochs, input_types);

        if (opts.use_rgb == 1)
            cdbn_layers{1}.n_map_v = 3;
        end

        % Runs the CDBN
        [cdbn_model, cdbn_layers] = run_CDBN(cdbn_layers, ...
                        train_data, train_data);

        output_cdbn_to_file(output_folder, i, cdbn_model, cdbn_layers);
    end
end

function [train_data, trainL, test_data, testL] = load_mnist_data(input_folder, i)
    input_filename = strcat(input_folder, '/fold_', num2str(i), '.mat');
    load(input_filename);
end

function output_cdbn_to_file(out_folder, i, cdbn_model, cdbn_layers)
    % For now I am using [2,2] hardcoded because this is always the size of
    % my CDBN
    cdbn_out = cell([2,2]);

    cdbn_out{1, 1} = cdbn_model{1,1}.W;
    cdbn_out{1, 2} = cdbn_model{1,1}.h_bias;
    cdbn_out{2, 1} = cdbn_model{1,2}.W;
    cdbn_out{2, 2} = cdbn_model{1,2}.h_bias;
    
    % TODO: Confirm that I don't need `cdbn_layers`
    
    if (~exist(out_folder, 'dir'))
        mkdir(out_folder);
    end
    
    output_filename = strcat(out_folder, '/cdbn_', num2str(i), '.mat');
    save(output_filename, 'cdbn_model');
end
