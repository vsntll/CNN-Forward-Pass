function outarray = applyfullconnect(inarray, filterbank, biasvals)
% Fully connected layer as 1x1xD2 output
% inarray: NxMxD1
% filterbank: NxMxD1xD2, biasvals: D2
[N, M, D1] = size(inarray);
[~, ~, ~, D2] = size(filterbank);
outarray = zeros(1, 1, D2);
for l = 1:D2
    filt = squeeze(filterbank(:, :, :, l));
    outarray(1, 1, l) = sum(inarray(:) .* filt(:)) + biasvals(l);
end
end
