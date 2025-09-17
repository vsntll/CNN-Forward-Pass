function outarray = applysoftmax(inarray)
% Softmax to a score vector
vals = squeeze(inarray);
vals = vals - max(vals); % For stability
expvals = exp(vals);
outarray = reshape(expvals / sum(expvals), [1, 1, length(vals)]);
end
