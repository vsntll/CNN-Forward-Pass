% Main routine for CNN forward pass
load('cifar10testdata.mat');      % loads imageset, trueclass, classlabels
load('CNNparameters.mat');        % loads all filters and biases for layers

Nimages = size(imageset, 4);
predicted_probs = zeros(Nimages, length(classlabels));
predicted_class = zeros(1, Nimages);

for idx = 1:Nimages
    x = imageset(:, :, :, idx);                  % NxMx3 uint8 image
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
    
    predicted_probs(idx, :) = squeeze(x);
    [~, predicted_class(idx)] = max(x(:));
end

% Calculate accuracy/confusion matrix
num_correct = sum(predicted_class == trueclass);
accuracy = num_correct / Nimages;
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

% Display confusion matrix
confmat = confusionmat(trueclass, predicted_class);
disp('Confusion Matrix:');
disp(confmat);
