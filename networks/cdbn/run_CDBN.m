function [model, layer] = run_CDBN(layer, train_data, test_data, old_model)
    tic;

    if ~exist('old_model', 'var')
        [model,layer] = cdbn2D(layer);
    else
        [model,layer] = cdbn2D(layer, old_model);
    end
    save('./model_parameter','model','layer');

    %CDBN_view(model, layer);
    toc;
end

function CDBN_view(model, layer)
    %% ---- Figure ---- %%

    % POOLING MAP 1
    figure(1);
    [r,c,n] = size(model{1}.output(:,:,:,1));
    visWeights(reshape(model{1}.output(:,:,:,1),r*c,n)); colormap gray
    title(sprintf('The first Pooling output'))
    drawnow
    
    % POOLING MAP 2
    figure(2);
    [r,c,n] = size(model{2}.output(:,:,:,1));
    visWeights(reshape(model{2}.output(:,:,:,1),r*c,n)); colormap gray
    title(sprintf('The second Pooling output'))
    drawnow

    % ORIGINAL SAMPLE
    figure(3);
    imagesc(layer{1}.inputdata(:,:,:,1)); colormap gray; axis image; axis off
    title(sprintf('Original Sample'));
end