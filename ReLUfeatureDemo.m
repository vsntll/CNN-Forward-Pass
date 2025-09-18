% Load the sample test image from local directory
load('cifar10testdata.mat'); % assumes variable 'testimages' of size HxWx3xN
sampleImg = imageset(:, :, :, 1);; % Use first test image, unnormalized for now

% Normalize input image (output same size)
normImg = applyimnormalize(sampleImg);

% Load filterbanks and bias vectors for CNN layers
params = load('CNNparameters.mat'); 
filterbanks = params.filterbanks;
biasvectors = params.biasvectors;

% Layer 1: Convolution (pass filterbank and bias vector)
conv1Out = applyconvolve(normImg, filterbanks{2}, biasvectors{2});

% ReLU activation
relu1Out = applyrelu(conv1Out);

% Layer 2: Max pooling
pool2Out = applymaxpool(relu1Out);

% Layer 3: Convolution + ReLU
conv3Out = applyconvolve(pool2Out, filterbanks{4}, biasvectors{4});
relu3Out = applyrelu(conv3Out);

% Visualize intermediate layer feature maps (relu3Out)
numFeatures = size(relu3Out, 3);
nCols = ceil(sqrt(numFeatures));
nRows = ceil(numFeatures / nCols);

figure('Name', 'Layer 3 ReLU Feature Maps');
for i = 1:numFeatures
    subplot(nRows, nCols, i);
    imshow(mat2gray(relu3Out(:,:,i)));
    title(sprintf('Feature %d', i));
end

% Show dummy softmax output heatmap for demonstration (10 classes)
dummySoftmax = rand(10,1);
dummySoftmax = dummySoftmax / sum(dummySoftmax);

figure('Name', 'Dummy Softmax Output Heatmap');
imagesc(dummySoftmax);
colorbar;
title('Dummy Softmax Output Heatmap');
xlabel('Class Index');
yticks([]);

% Print message in console
fprintf('Demo completed: Intermediate CNN features displayed and dummy softmax heatmap shown.\n');