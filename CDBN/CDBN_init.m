function layer = CDBN_init(train_data, filter_sizes, pooling_sizes, ...
                            n_epochs, input_types)
    %% ---- INITIALIZE THE PARAMETERS OF THE NETWORK ---- %%

    % FIRST LAYER SETTING
    layer{1} = default_layer2D();  % DEFAULT PARAMETERS SETTING, 

    % YOU CAN CHANGE THE PARAMETERS IN THE FOLLOWING LINES

    layer{1}.inputdata      = train_data;
    layer{1}.n_map_v        = 1;
    layer{1}.n_map_h        = 9;
    layer{1}.s_filter       = filter_sizes{1};
    layer{1}.stride         = [1 1];  
    layer{1}.s_pool         = pooling_sizes{1};
%     layer{1}.s_filter       = [11 1];
%     layer{1}.stride         = [1 1];  
%     layer{1}.s_pool         = [2 1];
    layer{1}.n_epoch        = n_epochs(1);
    layer{1}.learning_rate  = 0.05;
    layer{1}.sparsity       = 0.03;
    layer{1}.lambda1        = 5;
    layer{1}.lambda2        = 0.05;
    layer{1}.momentum       = 0;
    layer{1}.whiten         = 0;
    layer{1}.type_input     = input_types{1}; % OR 'Gaussian' 'Binary'

    % SECOND LAYER SETTING
    layer{2} = default_layer2D();  % DEFAULT PARAMETERS SETTING, 

    % YOU CAN CHANGE THE PARAMETERS IN THE FOLLOWING LINES

    layer{2}.n_map_v        = 9;
    layer{2}.n_map_h        = 16;
    layer{2}.s_filter       = filter_sizes{2};
    layer{2}.stride         = [1 1];
    layer{2}.s_pool         = pooling_sizes{2};
%     layer{2}.s_filter       = [2 1];
%     layer{2}.stride         = [1 1];
%     layer{2}.s_pool         = [2 1];
    layer{2}.n_epoch        = n_epochs(2);
    layer{2}.learning_rate  = 0.05;
    layer{2}.sparsity       = 0.02;
    layer{2}.lambda1        = 5;
    layer{2}.lambda2        = 0.05;
    layer{2}.momentum       = 0;
    layer{2}.whiten         = 0;
    layer{2}.type_input     = input_types{2};

    % THIRD LAYER SETTING ...   % YOU CAN CONTINUE TO SET THE THIRD, FOURTH,
    % FIFTH LAYERS' PARAMETERS WITH THE SAME STRUCT
    % MENTIONED ABOVE
end

