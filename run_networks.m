%clear all;

% Set parameters

disp(['Start time: ' datestr(now)]);

input_file_name = './mnist.mat';

opts.intercalations = 5;
opts.cnn_training_size = 10000;
%[cnn0, mode0_errs] = run_networks_mnist(input_file_name, 0, 7, opts);
%[cnn1, mode1_errs] = run_networks_mnist(input_file_name, 1, 7, opts);
%[cnn3, mode3_errs] = run_networks_mnist(input_file_name, 3, 7, opts);
%[cnn4, mode4_errs] = run_networks_mnist(input_file_name, 4, 7, opts);

dataset_folder = './yahoo_dataset/A1Benchmark';
run_networks_yahoo(dataset_folder, 1);

% Print current time
disp(['End time: ' datestr(now)]);
