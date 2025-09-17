function outarray = applyrelu(inarray)
% Applies ReLU activation function
% inarray: NxMxD
outarray = max(inarray, 0);
end
