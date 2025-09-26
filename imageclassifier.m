classify_unknown_images('ImagePool')

function classify_unknown_images(folder_path)
    %Classify all images within the specified folder with an unknown threshold
    %folder_path: Path to the folder containing images

    persistent filterbanks biasvectors layertypes
    if isempty(filterbanks) || isempty(biasvectors) || isempty(layertypes)
        load('CNNparameters.mat', 'filterbanks', 'biasvectors', 'layertypes');
    end

    image_files = dir(fullfile(folder_path, '*.*'));
    image_files = image_files(~[image_files.isdir]);

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'};
    image_files = image_files(arrayfun(@(f) any(strcmpi(valid_exts, lower(getExtension(f.name)))), image_files));

    function ext = getExtension(filename)
        [~,~,ext] = fileparts(filename);
    end

    for k = 1:length(image_files)
        try
            fprintf('Processing image %d/%d: %s\n', k, length(image_files), image_files(k).name);
            image_path = fullfile(folder_path, image_files(k).name);

            im_rgb = imread(image_path);
            im_rgb_resized = imresize(im_rgb, [32 32]);

            im_norm = applyimnormalize(im_rgb_resized);
            cur_layer = im_norm;
            for layer_idx = 1:length(layertypes)
                switch layertypes{layer_idx}
                    case 'imnormalize'
                        % already normalized
                    case 'convolve'
                        cur_layer = applyconvolve(cur_layer, filterbanks{layer_idx}, biasvectors{layer_idx});
                    case 'relu'
                        cur_layer = applyrelu(cur_layer);
                    case 'maxpool'
                        cur_layer = applymaxpool(cur_layer);
                    case 'fullconnect'
                        cur_layer = applyfullconnect(cur_layer, filterbanks{layer_idx}, biasvectors{layer_idx});
                    case 'softmax'
                        cur_layer = applysoftmax(cur_layer);
                end
            end

            probabilities = squeeze(cur_layer);
            [max_prob, class_idx] = max(probabilities);
            
            
            fprintf('Image %s classified as class %d (prob=%.3f)\n', image_files(k).name, class_idx, max_prob);
        catch ME
            fprintf('Error processing image %s: %s\n', image_files(k).name, ME.message);
        end
    end
end

