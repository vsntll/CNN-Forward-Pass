% Main routine for CNN forward pass
load('cifar10testdata.mat');      % loads imageset, trueclass, classlabels
load('CNNparameters.mat');        % loads all filters and biases for layers

Nimages = size(imageset, 4);
predicted_probs = zeros(Nimages, length(classlabels));
predicted_class = zeros(1, Nimages);

for idx = 1:Nimages
    x = imageset(:, :, :, idx);                  % NxMx3 uint8 image
    x = applyimnormalize(x);
    
    % Example for two layers; expand sequence for all 18 layers
    x = applyconvolve(x, conv1_filters, conv1_bias); % Layer 1
    x = applyrelu(x);                                % Layer 2
    x = applymaxpool(x);                             % Layer 3
    % if more layers add here
    
    % When at final fully connected + softmax stage:
    x = applyfullconnect(x, fc_filters, fc_bias);
    x = applysoftmax(x);
    
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
