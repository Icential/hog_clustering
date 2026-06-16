library(keras3)
library(ggplot2)
library(ggimage)
library(cluster)
library(uwot)
library(factoextra)
library(magick)
library(imager)
library(gapminder)
library(reticulate)
use_condaenv("base", required = TRUE)

# set.seed(42)

# parameters
p = 160 # principal components. 160 accounts for 95% of variance

# load test cifar10 dataset
cifar10 <- dataset_cifar10()
x <- cifar10$test$x
n <- dim(x)[1]

# create pca for every sample
df_raw <- matrix(NA, nrow = n, ncol = 32*32)
for (i in 1:n) {
  s <- as.cimg(x[i,,,])
  s <- imrotate(s, 90)
  s <- grayscale(s, method = "Luma", drop = TRUE)
  df[i, ] <- as.vector(s)
}
pca_raw <- prcomp(df_raw, center=TRUE, scale. = TRUE)
pca_raw_p <- pca_raw$x[,1:p]

# reconstruct a random image from p principal components
X_approx <- pca_raw$x[, 1:p] %*% t(pca_raw$rotation[, 1:p])
rand <- sample(1:10000, 2)
r1 <- matrix(X_approx[rand[1], ], nrow = 32, ncol = 32)
r1 <- as.cimg(r1)
plot(as.raster(r1, max = 255))

# make gif of reconstruction with increasing p
p_vals <- c(1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 120, 160, 256, 512, 1024)
img <- image_graph()
rand <- sample(1:10000, 10)
frames <- image_graph(800, 600, res=130)
for (p in p_vals) {
  # calculate reconstruction from scores
  X_approx <- pca_raw$x[, 1:p] %*% t(pca_raw$rotation[, 1:p])
  
  # put images to matrix form
  r1 <- matrix(X_approx[rand[1], ], nrow=32, ncol=32)
  r2 <- matrix(X_approx[rand[2], ], nrow=32, ncol=32)
  r3 <- matrix(X_approx[rand[3], ], nrow=32, ncol=32)
  r4 <- matrix(X_approx[rand[4], ], nrow=32, ncol=32)
  r5 <- matrix(X_approx[rand[5], ], nrow=32, ncol=32)
  r6 <- matrix(X_approx[rand[6], ], nrow=32, ncol=32)
  r7 <- matrix(X_approx[rand[7], ], nrow=32, ncol=32)
  r8 <- matrix(X_approx[rand[8], ], nrow=32, ncol=32)
  r9 <- matrix(X_approx[rand[9], ], nrow=32, ncol=32)
  r10 <- matrix(X_approx[rand[10], ], nrow=32, ncol=32)
  
  # matrix to image
  r1 <- as.cimg(r1)
  r2 <- as.cimg(r2)
  r3 <- as.cimg(r3)
  r4 <- as.cimg(r4)
  r5 <- as.cimg(r5)
  r6 <- as.cimg(r6)
  r7 <- as.cimg(r7)
  r8 <- as.cimg(r8)
  r9 <- as.cimg(r9)
  r10 <- as.cimg(r10)

  # plot and rasterize
  par(mar=c(0,0,0,0), oma=c(0,0,0,0), mfrow=c(2,5))
  plot(as.raster(r1, max=255))
  plot(as.raster(r2, max=255))
  plot(as.raster(r3, max=255))
  plot(as.raster(r4, max=255))
  plot(as.raster(r5, max=255))
  plot(as.raster(r6, max=255))
  plot(as.raster(r7, max=255))
  plot(as.raster(r8, max=255))
  plot(as.raster(r9, max=255))
  plot(as.raster(r10, max=255))
}
dev.off()

gif <- image_animate(frames, fps=4, optimize=TRUE)
image_write(gif, "vis/pca_reconstruction.gif")
