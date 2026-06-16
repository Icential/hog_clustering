library(keras3)
library(ggplot2)
library(cluster)
library(uwot)
library(factoextra)
library(OpenImageR)
library(reticulate)
use_condaenv("base", required = TRUE)

# set.seed(42)

# p principal components and k clusters
p = 50
k = 10
c = 12 # how many cells (increase, more detailed)
o = 8 # how many gradient directions (increase, more detailed)

# load test cifar10 dataset
cifar10 <- dataset_cifar10()
x <- cifar10$test$x
n <- dim(x)[1]

# create pca for every sample's HOG
df_hog <- matrix(NA, nrow = n, ncol = length(HOG(x[1,,,], cells=c, orientations=o)))
for (i in 1:n) {
  s <- HOG(x[i,,,], cells=c, orientations=o)
  df_hog[i, ] <- s
}
pca_hog <- prcomp(df_hog, center=TRUE, scale. = TRUE)
pca_hog_p <- pca_hog$x[,1:p]

# plot distribution of singular values variances
vars_hog <- cumsum(pca_hog$sdev^2) / sum(pca_hog$sdev^2)
plot(vars_hog, type="l", xlab = "# of principal components", ylab = "Proportion of variance")
which(vars_hog >= 0.95)[1] # how many principal components enough to explain 95% of variance

# plot cumulative pc plot for raw and hog
plot(1:1024, vars_raw, type="l", col="red", lwd=2, xlab="Number of Principal Components", ylab="Proportion of Variance")
lines(1:1024, vars_hog[1:1024], col="green", lwd=2)
legend("bottomright", legend=c("Raw Pixels", "Histogram of Gradients"), col=c("red", "green"), pch=c(20, 20))

# calculating best k-means silhouette score for iterative p
scores_hog <- vector("list", p)
ks_hog <- vector("list", p)
for (i in 1:p) {
  best_score = -999999
  pca_hog_p = pca_hog$x[,1:i]
  for (j in 2:10) {
    cluster <- kmeans(x=pca_hog_p, centers=j, iter.max = 100, nstart = 10)
    sil <- mean(silhouette(cluster$cluster, dist(pca_hog_p))[, 3]) # calculate silhouette score
    if (sil > best_score) {
      best_score <- sil
      scores_hog[i] <- sil
      ks_hog[i] <- j
    }
  }
  print(
    paste("Best silhouette score for", i, "components:", best_score,
      "at", ks_hog[i], "clusters."
    )
  )
}

# plot differences between principal components of raw and hog
plot(1:20, scores_raw[1:20], type="l", col="red", lwd=2, xlab="Principal Components", ylab="Silhouette Score", main="Best Silhouette Score For Different Amount of Principal Components", ylim=c(0, 0.6))
lines(1:20, scores_hog[1:20], col="green", lwd=2)
legend("topright", legend=c("Raw Pixels", "Histogram of Gradients"), col=c("red", "green"), pch=c(20, 20))

# plot number of clusters used
plot(1:20, ks_raw[1:20], type="l", col="red", lwd=2, xlab="Number of Clusters", ylab="K", main="Best K for K-means Clustering", ylim=c(2, 4))
lines(1:20, ks_hog[1:20], col="green", lwd=2)
legend("topright", legend=c("Raw Pixels", "Histogram of Gradients"), col=c("red", "green"), pch=c(20, 20))

# UMAP visualization
km = kmeans(x=pca_hog$x[,1:10], centers=3, iter.max = 100, nstart = 10)
vis <- umap(pca_hog$x[,1:10], n_components=2)
vis_df <- data.frame(x=vis[, 1], y=vis[,2], cluster=as.factor(km$cluster))
ggplot(vis_df, aes(x, y, color=cluster)) + geom_point(alpha=0.4, size=0.8) + theme_minimal() + labs(title="UMAP of HOG PCA at p=10")

# Comparing different k clusters with silhouette score
fviz_nbclust(pca_hog_p, kmeans, method="silhouette", k.max=20) + geom_vline(xintercept = 3, linetype=2)

