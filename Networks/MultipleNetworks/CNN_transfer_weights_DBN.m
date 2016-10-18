function cnn = CNN_transfer_weights_DBN(cnn, weights, sizes)
    % In the following code:
    % "r" is the input matrix
    % "c" feature map
    % "a" is the matrix that is being created, that, when convolved with r,
    %       produces something similar to "c".

    W1 = weights{1};
    W2 = weights{2};
    
    % TODO: make the sizes generic. For now, I will suppose I know the
    % sizes of the convolutional layers.
    size_mnist_img = [28,28];
    size_after_first_pooling = [11,11];
    size_a1 = [7,7];
    size_a2 = [6,6];
    n_filters1 = 9;
    n_filters2 = 16;
    
    a1 = zeros([size_a1 n_filters1]);
    a2 = zeros([size_a2 n_filters2]);

    
    size_c1 = [ (size_mnist_img(1) - size_a1(1) + 1), (size_mnist_img(2) - size_a1(2) + 1) ];
    
    % I use 11 because it will be the size of the feature map after pooling
    size_c2 = [ (size_after_first_pooling(1) - size_a2(1) + 1), (size_after_first_pooling(2) - size_a2(2) + 1) ];

    for filter = 1:n_filters1
        for a_r = 1:size_a1(1)
            for a_c = 1:size_a1(2)
                temp = 0;
                offset = calculate_offset(a_r, a_c, size_mnist_img);
                for c_r = 1:size_c1(1)
                    for c_c = 1:size_c1(2)
                        [w_r, w_c] = calculate_position(filter, a_r, a_c, c_r, c_c, offset, size_c1, size_mnist_img);
                        temp = temp + W1(w_r, w_c);
                        %fprintf('a(%d, %d) += W(%d, %d), i.e., %f [c(%d, %d)]\n', a_r, a_c, w_r, w_c, temp, c_r, c_c);
                    end
                end
                a1(a_r, a_c, filter) = a1(a_r, a_c, filter) + temp / prod(size_c1);
            end
        end
        %fprintf('STARTING NEW FILTER\n');
    end

    
    for filter = 1:n_filters2
        for a_r = 1:size_a2(1)
            for a_c = 1:size_a2(2)
                temp = 0;
                offset = calculate_offset(a_r, a_c, size_mnist_img);
                for c_r = 1:size_c2(1)
                    for c_c = 1:size_c2(2)
                        [w_r, w_c] = calculate_position(filter, a_r, a_c, c_r, c_c, offset, size_c2, size_after_first_pooling);
                        temp = temp + W2(w_r, w_c);
                        %fprintf('a(%d, %d) += W(%d, %d), i.e., %f [c(%d, %d)]\n', a_r, a_c, w_r, w_c, temp, c_r, c_c);
                    end
                end
                a2(a_r, a_c, filter) = a2(a_r, a_c, filter) + temp / prod(size_c2);
            end
        end
    end

    blah;
    
end

function [w_r, w_c] = calculate_position(filter, a_r, a_c, c_r, c_c, offset, size_c, size_r)
    % a_r is the row in a
    % a_c is the column in a
    % c_r is the row in c
    % c_c is the column in c
    % skip is how many positions must be skipped in a row in "r"
    % offset is how much must be skipped in the beginning of "r"
    w_r = offset + c_c + size_r(2) * (c_r - 1);
    w_c = (filter - 1) * prod(size_c) + (c_r - 1) * size_c(2) + c_c;
end

function offset = calculate_offset(a_r, a_c, size_r)
    offset = a_c + (size_r(2) * (a_r - 1)) - 1;
end
