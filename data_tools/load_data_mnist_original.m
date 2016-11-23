% Strongly based on:
% https://github.com/lonl/CDBN/blob/master/DemoCDBN_Binary_2D.m
function [train_data, train_labels, test_data, test_labels] = ...
                    load_data_mnist_original(input_file_name, add_noise)
    % LOAD MNIST DATA TO TEST BINARY CDBN

    % The difference between this function and `load_data()` is that it
    % does not mix the `test_data` and `test_labels` into the same place.
    % This allows me to use the original train and test datasets (to see if
    % the results will be consistent with the rest of the results I have
    % been getting -- they should, but God knows if I am not doing anything
    % wrong =) ) 
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
    
    train_data(:,:,:,50001:60000) = val_data;
    train_labels(50001:60000) = val_labels;
    
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

