% Strongly based on:
% https://github.com/lonl/CDBN/blob/master/DemoCDBN_Binary_2D.m
function [data_x, data_y] = load_data(input_file_name, add_noise)
	% LOAD MNIST DATA TO TEST THE BINARY CDBN

	%load ./data/mnist/mnist.mat;
	load(input_file_name);
	train_data     = cell2mat(mnist_data(1,1));
	train_data     = double(reshape(train_data', [28,28,1,50000]));
	train_labels   = cell2mat(mnist_data(1,2));
    
	val_data       = cell2mat(mnist_data(2,1));
	val_data       = double(reshape(val_data', [28,28,1,10000]));
	val_labels     = cell2mat(mnist_data(2,2));
    
	test_data      = cell2mat(mnist_data(3,1));
	test_data      = double(reshape(test_data', [28,28,1,10000]));
	test_labels    = cell2mat(mnist_data(3,2));
    
	data_x = train_data;
	data_x(:,:,:,50001:60000) = val_data;
	data_x(:,:,:,60001:70000) = test_data;

	data_y = train_labels;
	data_y(50001:60000) = val_labels;
	data_y(60001:70000) = test_labels;

	% REMOVES PART OF THE INPUT, IF NECESSARY
	% (you can change the 1:50000 into any smaller number, as needed)

% 	train_data     = train_data(:,:,:,1:60000);
% 	test_data      = test_data(:,:,:,1:10000);
% 	trainL         = train_labels(1:60000);
% 	testL          = test_labels(1:10000);

	% ADD NOISE
	if add_noise
	    fprintf('--------------- ADD NOISE IN TEST DATA -------------- \n');
	    b          = rand(size(test_data)) > 0.9;
	    noised     = test_data;
	    rnd        = rand(size(test_data));
	    noised(b)  = rnd(b);
	    test_data  = noised;
	end
end
