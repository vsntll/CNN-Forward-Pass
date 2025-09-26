function classify_unknown_images(folder_path, threshold)
    %Classify all images within the specified folder with an unknown threshold
    %folder_path: Path to the folder containing images

    % Load CNN parameters (persistent to do this once since they dont change)
    persistent filterbanks biasvectors layertypes
    if isempty(filterbanks) || isempty(biasvectors) || isempty(layertypes)
        load('CNNparameters.mat', 'filterbanks', 'biasvectors', 'layertypes');
    end

    %list all image files in the folder
    image_files = dir(fullfile(folder_path, '*.*'));
    image_files = image_files(~[image_files.isdir]);

    %Filter for image extensions so it doesnt matter whats in the folder
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'};
    % Only include files with the valid extensions
    image_files = image_files(arrayfun(@(f) any(strcmpi(valid_exts, lower(fileparts(f.name)))), image_files));

    %! Oh boy....
    for k = 1:length(image_files)
        image_name = image_files(k).name;
        image_path = fullfile(folder_path, image_name);

        % Read and resize image to the specs in desc. (32x32x3)
        im_rgb = imread(image_path)
        im_rgb_resized = imresize(im_rgb, [32 32]);

        % Normalize
        im_norm = applyimnormalize(im_rgb_resized);


        % FW pass through the CNN layers
        cur_layer = im_norm
        for layer_idx = 1:length(layertypes)
            switch layertypes{layer_idx}
                case 'imnormalize'
                    cur_layer = cur_layer %already normalized doesn't make sense
                case 'convolve'
                    cur_layer = applyconvolve(cur_layer, filterbanks{layer_idx}, biasvectors{layers_idx});
                case 'relu'
                    cur_layer = applyrelu(cur_layer);
                case 'maxpool'
                    cur_layer = maxpool(cur_layer);
                case 'fullconnect'
                    cur_layer = applyfullconnect(cur_layer, filterbanks{layer_idx}, biasvectors{layer_idx});
                case 'softmax'
                    cur_layer = applysoftmax(cur_layer);
                otherwise
                    error('Unknown layer type: %s', layertypes{layer_idx});
            end
        end
        
        % Output probabilities vector (1x1x10);
        probabilities = squeeze(cur_layer);

        % Classification
        [max_prob, class_idx] = max(probabilities);
        if max_prob < threshold
            fprintf('Image %s classified as UNKNOWN with max probability %.3f\n', image_name, max_prob);
        else 
            fprintf('Image %s classified as class %d with probability %d, %.3f\n', image_name, class_idc, max_prob);
        end
    end
end


    
    
    
    
    
    
    end