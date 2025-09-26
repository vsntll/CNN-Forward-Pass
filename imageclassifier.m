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

        