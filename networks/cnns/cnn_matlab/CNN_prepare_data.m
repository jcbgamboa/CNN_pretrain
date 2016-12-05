function [train_out, trainL_out, test_out, testL_out] = ...
                    CNN_prepare_data(train_data, trainL, test_data, testL)
    % The inputs train_data and test_data are in a format that is
    % appropriate for the CDBN (and not for the DeepLearnToolbox). This
    % function "preprocesses" it so that it can be used by in the CNNs from
    % the DeepLearnToolbox.

    % [this part it turned out to be very easy to do it, apparently]
    train_out = squeeze(train_data);
    test_out  = squeeze(test_data);
    
    % While, for CDBN, the labels were just a number, for this CNN the
    % labels are a vector with 10 values (each value indicates one of the
    % digits, and can only contain either 0 or 1 -- only one of the values
    % is 1 for a given image).
    trainL_out = labelmatrix(trainL, 10)';
    testL_out  = labelmatrix(testL,  10)';
end