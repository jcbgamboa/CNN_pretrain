%clear all;

% Set parameters

disp(['Start time: ' datestr(now)]);

input_file_name = './mnist.mat';

opts.intercalations = 5;
%[cnn0, mode0_errs, losses0] = run_networks_mnist(input_file_name, 0, 7, opts);
%[cnn1, mode1_errs, losses1] = run_networks_mnist(input_file_name, 1, 7, opts);

%[cnn2, mode2_errs, losses2] = run_networks_mnist(input_file_name, 1, 7, opts);
%[cnn3, mode3_errs, losses3] = run_networks_mnist(input_file_name, 3, 7, opts);
%[cnn4, mode4_errs, losses4] = run_networks_mnist(input_file_name, 4, 7, opts);

% [cnn0_full, mode0_errs_full, losses0_full] = run_networks_mnist(input_file_name, 0, 1, opts);
% opts.cdbn_unit_type = 'Binary';
% [cnn1_full, mode1_errs_full, losses1_full] = run_networks_mnist(input_file_name, 1, 1, opts);
% opts.cdbn_unit_type = 'Gaussian';
% [cnn2_full, mode2_errs_full, losses2_full] = run_networks_mnist(input_file_name, 1, 1, opts);
% [cnn3_full, mode3_errs_full, losses3_full] = run_networks_mnist(input_file_name, 3, 1, opts);
% %[cnn4, mode4_errs, losses] = run_networks_mnist(input_file_name, 4, 1, opts);
% 
% opts.cnn_training_size = 10000;
% [cnn0_mini, mode0_errs_mini, losses0_mini] = run_networks_mnist(input_file_name, 0, 1, opts);
% opts.cdbn_unit_type = 'Binary';
% [cnn1_mini, mode1_errs_mini, losses1_mini] = run_networks_mnist(input_file_name, 1, 1, opts);
% opts.cdbn_unit_type = 'Gaussian';
% [cnn2_mini, mode2_errs_mini, losses2_mini] = run_networks_mnist(input_file_name, 1, 1, opts);
% [cnn3_mini, mode3_errs_mini, losses3_mini] = run_networks_mnist(input_file_name, 3, 1, opts);

%dataset_folder = './yahoo_dataset/A1Benchmark';
%run_networks_yahoo(dataset_folder, 1);

% First I want to call the CDBN function again with the cross-validation
opts.normalize = 0;
opts.use_rgb = 0;
opts.cdbn_unit_type = 'Binary';
run_cdbn_mnist(7, './mnist_folds', './pretrained_weights/cdbn/mnist/binary', opts);

opts.cdbn_unit_type = 'Gaussian';
run_cdbn_mnist(7, './mnist_folds', './pretrained_weights/cdbn/mnist/gaussian', opts);

% -----
% Abuses of the MNIST function to run CIFAR
opts.normalize = 1;
opts.use_rgb = 1;
opts.cdbn_unit_type = 'Binary';
run_cdbn_mnist(6, './cifar_folds', './pretrained_weights/cdbn/cifar/binary', opts);

opts.cdbn_unit_type = 'Gaussian';
run_cdbn_mnist(6, './cifar_folds', './pretrained_weights/cdbn/cifar/gaussian', opts);

% Print current time
disp(['End time: ' datestr(now)]);

