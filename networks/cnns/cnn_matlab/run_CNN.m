function cnn = run_CNN(train_data, trainL, test_data, testL, opts, model)
    % Loads the data from the original dataset (this is not needed, and is
    % here only for reference -- in case I want to compare, e.g., `train_x`
    % with `train_data`.
    %load('C:\project\DeepLearnToolbox\data\mnist_uint8.mat')
    %train_x = double(reshape(train_x',28,28,60000))/255;
    %test_x = double(reshape(test_x',28,28,10000))/255;
    %train_y = double(train_y');
    %test_y = double(test_y');



    %view_CNN(cnn);
end

function view_CNN(cnn)
    count = 1;
    for i = 1:9
        for j = 1:16
            subplot(9,16,count);
            imshow(cnn.layers{4}.k{i}{j} - min(min(cnn.layers{4}.k{i}{j})));
            %subplot(3,3,i);
            %imshow(cnn.layers{2}.k{1}{i} - min(min(cnn.layers{2}.k{1}{i})));
        end
        count = count + 1;
    end
end
