load('cifar10testdata.mat');      % loads imageset, trueclass, classlabels
load('CNNparameters.mat');        % loads all filters and biases for layers

x=imageset(:, :, :, 1);

x = applyimnormalize(x);    %Layer 1
x = applyconvolve(x, filterbanks{2}, biasvectors{2}); % Layer 2

figure;
for l = 1:10
    subplot(2, 5, l);              % 2 rows x 5 columns for 10 filters
    imshow(x(:, :, l), []); % [] automatically scales the display
    s = size(x(:, :, 1));   %records the size of each sub image
    title(['Image ' num2str(l) ' size: ' num2str(s(1)) ' x ' num2str(s(2))]); %displays the image and its size
end
