function dbn = run_DBN(train_data, trainL, test_data, testL)
    %train dbn
    %dbn.sizes = [3600 576];
    dbn.sizes = [1000 20];
    opts.numepochs =   100;
    opts.batchsize =   100;
    opts.momentum  =   [0.5, 0.9];
    opts.alpha     =   0.01;
    opts.lambda    =   0.05;
    opts.sparsity  =   0.02;
    dbn = dbnsetup(dbn, train_data, opts);
    dbn = dbntrain(dbn, train_data, opts);
end

