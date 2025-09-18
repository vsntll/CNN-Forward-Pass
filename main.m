% Main routine for CNN forward pass
load('cifar10testdata.mat');      % loads imageset, trueclass, classlabels
load('CNNparameters.mat');        % loads all filters and biases for layers

Nimages = size(imageset, 4);
predicted_probs = zeros(Nimages, length(classlabels));
predicted_class = zeros(1, Nimages);

for d = 1:length(layertypes)
    x = imageset(:, :, :, d);                  % NxMx3 uint8 image
    x = applyimnormalize(x);                     % Layer 1
    
    % Example for two layers; expand sequence for all 18 layers
    x = applyconvolve(x, filterbanks{2}, biasvectors{2}); % Layer 2
    x = applyrelu(x);                                % Layer 3
    x = applyconvolve(x, filterbanks{4}, biasvectors{4}); % Layer 4
    x = applyrelu(x);                                % Layer 5
    x = applymaxpool(x);                             % Layer 6
    x = applyconvolve(x, filterbanks{7}, biasvectors{7}); % Layer 7
    x = applyrelu(x);                                % Layer 8
    x = applyconvolve(x, filterbanks{9}, biasvectors{9}); % Layer 9
    x = applyrelu(x);                                % Layer 10
    x = applymaxpool(x);                             % Layer 11
    x = applyconvolve(x, filterbanks{12}, biasvectors{12}); % Layer 12
    x = applyrelu(x);                                % Layer 13
    x = applyconvolve(x, filterbanks{14}, biasvectors{14}); % Layer 14
    x = applyrelu(x);                                % Layer 15
    x = applymaxpool(x);                             % Layer 16
    x = applyfullconnect(x, filterbanks{17}, biasvectors{17});  % Layer 17
    x = applysoftmax(x);                             % Layer 18
    
   fprintf('layer %d is of type %s\n',d,layertypes{d});
    filterbank = filterbanks{d};
    if not(isempty(filterbank))
        fprintf('filterbank size %d x %d x %d x %d\n', ...
            size(filterbank,1),size(filterbank,2), ...
            size(filterbank,3),size(filterbank,4));
        biasvec = biasvectors{d};
        fprintf(' number of biases is %d\n',length(biasvec));
    end
end

%loading this file defines imrgb and layerResults
load 'debuggingTest.mat'
%sample code to show image and access expected results
figure; imagesc(imrgb); truesize(gcf,[64 64]);
for d = 1:length(layerResults)
    result = layerResults{d};
    fprintf('layer %d output is size %d x %d x %d\n',...
        d,size(result,1),size(result,2), size(result,3));
end
%find most probable class
classprobvec = squeeze(layerResults{end});
[maxprob,maxclass] = max(classprobvec);
%note, classlabels is defined in ’cifar10testdata.mat’
fprintf('estimated class is %s with probability %.4f\n',...
    classlabels{maxclass},maxprob);
