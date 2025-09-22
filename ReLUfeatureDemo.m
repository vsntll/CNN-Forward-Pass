% demo.m -- CNN demonstration for CMPENEE 454 Project 1
% Loads the sample image and parameters and walks through the 18-layer pipeline

load('debuggingTest.mat');   % Loads sample image imrgb
load('CNNparameters.mat');   % Loads layertypes, filterbanks, biasvectors
load('cifar10testdata.mat'); % Loads classlabels

figure; imagesc(imrgb); truesize(gcf, [64 64]);
title('Input Image'); drawnow;

data = imrgb;
for d = 1:length(layertypes)
    switch layertypes{d}
        case 'imnormalize'
            data = applyimnormalize(data);
        case 'convolve'
            data = applyconvolve(data, filterbanks{d}, biasvectors{d});
        case 'relu'
            data = applyrelu(data);
        case 'maxpool'
            data = applymaxpool(data);
        case 'fullconnect'
            data = applyfullconnect(data, filterbanks{d}, biasvectors{d});
        case 'softmax'
            data = applysoftmax(data);
    end
    if ismember(layertypes{d}, {'convolve', 'relu', 'maxpool'})
        sz = size(data);
        numChannels = sz(3);
        colMax = ceil(sqrt(numChannels));
        rowMax = ceil(numChannels / colMax);
        figure;
        for ch = 1:numChannels
            subplot(rowMax, colMax, ch);
            imagesc(data(:,:,ch)); colormap gray; axis off;
        end
        sgtitle(sprintf('Layer %d: %s', d, layertypes{d}));
    end
end

finalProbs = squeeze(data);
figure;
bar(finalProbs);
title('Predicted Class Probabilities'); 
set(gca, 'XTickLabel', classlabels);
ylabel('Probability');

[maxprob, maxclass] = max(finalProbs);
disp(['Predicted class: ' classlabels{maxclass} ', Probability: ' num2str(maxprob)]);
