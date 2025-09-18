function outarray = applymaxpool(inarray)
% Applies 2x2 max pooling
% inarray: 2Nx2MxD, outarray: NxMxD
[N2, M2, D] = size(inarray);
N=N2 / 2;
M=M2 / 2;
outarray = zeros(N,M,D);
for k = 1:D
    for i = 1:N
        for j = 1:M
            block = inarray(2*i-1:2*i, 2*j-1:2*j, k);
            outarray(i, j, k) = max(block(:));
        end
    end
end
end