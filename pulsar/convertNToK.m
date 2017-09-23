function X_K = convertNToK(X_N, K)
	[U S] = pca(X_N);
	X_K = X_N * U(:, 1:K);
end
