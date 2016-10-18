function [err, bad] = cnn_regression_test(net, x, y, opts)
    % Calculates the error when the network is calculating a regression

    net = cnnff(net, x, opts);

    err = net.o - y;

    % Multiplying the sign of x with x is the same as doing |x|, i.e.,
    % taking the magnitude of x. In this case, we want to get all elements
    % of `err` whose magnitude is bigger than a threshold.
    bad = (sign(err) .* err) > 1;
end
