function outarray = applyimnormalize(inarray)
    outarray = double(inarray)/255.0-0.5;
end