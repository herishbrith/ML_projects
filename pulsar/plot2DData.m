function [] = plot2DData(X, y, centroids)

plot(X(y==0,1), X(y==0,2), 'bo');
hold on;

if ~isempty(centroids)
	plot(centroids(1,1), centroids(1,2), "rx");
	plot(centroids(2,1), centroids(2,2), "bx");
end
plot(X(y==1,1), X(y==1,2), 'ro');
xlabel("z1");
ylabel("z2");
legend("y==0", "y==1");
hold off;

end
