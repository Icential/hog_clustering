library(keras3)
library(ggplot2)
library(cluster)
library(uwot)
library(factoextra)
library(gganimate)
library(imager)
library(reticulate)
use_condaenv("base", required = TRUE)

# set.seed(42)

# parameters
p = 160 # principal components. 160 accounts for 95% of variance
k = 10 # clusters

# load test cifar10 dataset
cifar10 <- dataset_cifar10()
x <- cifar10$test$x
n <- dim(x)[1]

# create pca for every sample
df_raw <- matrix(NA, nrow = n, ncol = 32*32)
for (i in 1:n) {
  s <- as.cimg(x[i,,,])
  s <- imrotate(s, 90) # for some reason the image is read as rotated
  s <- grayscale(s, method = "Luma", drop = TRUE)
  df_raw[i, ] <- as.vector(s)
}
pca_raw <- prcomp(df_raw, center=TRUE, scale. = TRUE)
pca_raw_p <- pca_raw$x[,1:p]

# plot distribution of singular values variances
vars_raw <- cumsum(pca_raw$sdev^2) / sum(pca_raw$sdev^2)
plot(vars_raw, type="l", xlab = "# of principal components", ylab = "Proportion of variance")
which(vars_raw >= 0.95)[1] # how many principal components enough to explain 95% of variance

# calculating best k-means silhouette score for iterative p
scores_raw <- vector("list", p)
ks_raw <- vector("list", p)
for (i in 1:p) {
  best_score = -999999
  pca_raw_p = pca_raw$x[,1:i]
  for (j in 2:10) {
    cluster <- kmeans(x=pca_raw_p, centers=j, iter.max = 100, nstart = 10)
    sil <- mean(silhouette(cluster$cluster, dist(pca_raw_p))[, 3]) # calculate silhouette score
    if (sil > best_score) {
      best_score <- sil
      scores_raw[i] <- sil
      ks_raw[i] <- j
    }
  }
  print(
    paste("Best silhouette score for", i, "components:", best_score,
          "at", ks_raw[i], "clusters."
    )
  )
}
plot(1:p, scores_raw, type="l")

# UMAP visualization
km = kmeans(x=pca_raw$x[,1:10], centers=3, iter.max = 100, nstart = 10)
vis <- umap(pca_raw$x[,1:10], n_components=2)
vis_df <- data.frame(x=vis[, 1], y=vis[,2], cluster=as.factor(km$cluster))
ggplot(vis_df, aes(x, y, color=cluster)) + geom_point(alpha=0.4, size=0.8) + theme_minimal() + labs(title="UMAP of Raw Pixels PCA at p=10")

# Comparing different k clusters with silhouette score
fviz_nbclust(pca_raw_p, kmeans, method="silhouette", k.max=20) + geom_vline(xintercept = 3, linetype=2)
