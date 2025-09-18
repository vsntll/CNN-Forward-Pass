function outarray = applyconvolve(inarray, filterbank, biasvals)
% Applies convolution with filter bank and bias
% inarray: NxMxD1
% filterbank: RxCxD1xD2, biasvals: D2
[N, M, D1] = size(inarray);
[R, C, ~, D2] = size(filterbank);
outarray = zeros(N, M, D2);
for l = 1:D2
    temp = zeros(N, M);
    for k = 1:D1
        filt = squeeze(filterbank(:, :, k, l));
        temp = temp + imfilter(inarray(:, :, k), filt, 'conv', 'same');
    end
    outarray(:, :, l) = temp + biasvals(l);
end
end
