library(keras3)
library(ggplot2)
library(uwot)
library(imager)
library(reticulate)
use_condaenv("base", required = TRUE)

# load test cifar10 dataset
cifar10 <- dataset_cifar10()
x <- cifar10$test$x
cifar10_labels <- c("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")

# plot one random image sample
rand <- sample(1:1000, 1)
x_sample <- x[rand, , , ]
x_sample <- as.cimg(x_sample)
x_sample <- imrotate(x_sample, 90) # for some reason cimg rotates the img?
x_sample <- grayscale(x_sample, method = "Luma", drop = TRUE)
plot(as.raster(x_sample, max = 255))

# check distribution of y labels (seems all balanced)
y <- cifar10$test$y
df <- data.frame(val = y)
ggplot(df, aes(x = val)) + geom_histogram(binwidth=0.5)

# UMAP visualization (raw)
vis <- umap(pca_raw$x[,1:10], n_components=2)
vis_df <- data.frame(x=vis[, 1], y=vis[,2], label = as.factor(cifar10_labels[cifar10$test$y + 1]))
ggplot(vis_df, aes(x, y, color=label)) + geom_point(alpha=0.4, size=0.8) + theme_minimal() + labs(title="UMAP of Raw Pixels PCA at p=10 (CIFAR-10 Labels)")

# UMAP visualization (hog)
vis <- umap(pca_hog$x[,1:10], n_components=2)
vis_df <- data.frame(x=vis[, 1], y=vis[,2], label = as.factor(cifar10_labels[cifar10$test$y + 1]))
ggplot(vis_df, aes(x, y, color=label)) + geom_point(alpha=0.4, size=0.8) + theme_minimal() + labs(title="UMAP of HOG PCA at p=10 (CIFAR-10 Labels)")
