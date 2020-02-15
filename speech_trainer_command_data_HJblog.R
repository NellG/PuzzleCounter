# Making a smash up of two plog posts:
# Tensorflow Blog voice recognition post from here:
# https://blogs.rstudio.com/tensorflow/posts/2018-06-06-simple-audio-classification-keras/
# Spectrograms in R blog post from here:
# https://hansenjohnson.org/post/spectrograms-in-r/


library(stringr)
library(dplyr)
library(tfdatasets)
library(tuneR)
library(signal)
library(pbapply)
library(keras)
use_implementation('tensorflow')
library(tensorflow)

# if needed download speech commands audio data from
# http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
# and untar using the following command
#untar("data/speech_commands_v0.01.tar.gz", 
#      exdir="data/speech_commands_v0.01")

# from fs (file system operations) package borrow dir_ls
files <- fs::dir_ls(
  path = "data/speech_commands_v0.01/",
  recurse = TRUE,
  glob = "*.wav"
)

# keep all files except those with background_noise in the name
files <- files[!str_detect(files, "background_noise")]

# make a tibble of filenames, class (word spoken) of each
# file and class as an integer
df <- tibble(
  fname = files,
  class = fname %>% str_extract("1/.*/") %>%
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)

# set up training and test sets
set.seed(1)
ntrain <- 4000
ntest <- 300

all <- which(df$class_id < 3)
train <- sample(all, ntrain, replace=FALSE)
test <- sample(all[-train], ntest, replace=FALSE)

# set up spectrogram creation
window_size_ms <- 30 # window in ms
window_stride_ms <- 10 # step size between windows in ms
# note: files should have 16000 samples per second
window <- as.integer(16*window_size_ms) # samples per window
stride <- as.integer(16*window_stride_ms) # samples per stride
n_freq <- as.integer(2^(trunc(log(window, 2))))
n_chunks <- length(seq(window/2, 16000-window/2, stride))-1
overlap = window - stride

make_spec <- function(filename, window, overlap){
  data <- readWave(filename) # read in audio
  snd <- data@left # extract signal
  n <- as.integer(2^(trunc(log(window, 2)+1)))
  if (length(snd) < 16000) {
        snd <- c(snd, rep(0, 16000))[1:16000]
  }

  # create spectrogram and crop at max frequency
  spec <- specgram(x=snd, n=n, Fs=data@samp.rate,
                   window=window, overlap=overlap)
  P <- abs(spec$S)
  P <- log10(P+0.01) # normalize
  #print(paste('Row: ', nrow(P), ' Col: ', ncol(P), ' n: ', n))
  #print(paste('Max freq:', max(spec$f), ' Times: ', spec$t))
  spec_vec <- as.vector(t(P))
  # to turn the spec lists back into matrices use matrix(spec, ncol = n/2)
  return(spec_vec)
}

data_generator <- function(df, window, overlap){

  # set up parameters for spectrogram creation
  # window and overlap are in samples

  print(paste('Start processing', nrow(df), 'images'))
  df$spec <- pblapply(df$fname, make_spec, window=window, overlap=overlap)

  print('Success!')
  return(df)

}

train_data <- data_generator(df[train,], window, overlap)
test_data <- data_generator(df[test,], window, overlap)


test_img <- t(matrix(as.numeric(train_data$spec[[1]]), ncol=n_freq/2))
image(z=t(test_img), axes=F)

# Build training  and test arrays
train_array <- unlist(t(train_data$spec))
dim(train_array) <- c(n_chunks, n_freq, length(train_data$spec), 1)

test_array <- unlist(t(test_data$spec))
dim(test_array) <- c(n_chunks, n_freq, length(test_data$spec), 1)

# Reorder dimensions
train_array <- aperm(train_array, c(3, 1, 2, 4))
test_array <- aperm(test_array, c(3, 1, 2, 4))

image(z=(test_array[4,,,1]))

# Build convolutional neural network model

model <- keras_model_sequential()

# model specified by HJ post
# model %>%
#   layer_conv_2d(kernel_size=c(3,3), filter=32, activation='relu',
#                 padding='same', input_shape=c(n_chunks, n_freq, 1),
#                 data_format='channels_last') %>%
#   layer_conv_2d(kernel_size=c(3,3), filter=32, activation='relu',
#                 padding='valid') %>%
#   layer_max_pooling_2d(pool_size=2) %>%
#   layer_dropout(rate=0.25) %>%
# 
#   layer_conv_2d(kernel_size=c(3,3), filter=64, strides=2,
#                 activation='relu', padding='same') %>%
#   layer_conv_2d(kernel_size=c(3,3), filter=64, activation='relu',
#                 padding='valid') %>%
#   layer_max_pooling_2d(pool_size=2) %>%
#   layer_dropout(rate=0.25) %>%
# 
#   layer_flatten() %>%
#   layer_dense(units=50, activation='relu') %>%
#   layer_dropout(rate=0.25) %>%
#   layer_dense(units=1, activation='sigmoid')

# model specified by tensorflow post
model %>%
  layer_conv_2d(input_shape=c(n_chunks, n_freq,1), filters=32,
                        kernel_size=c(3,3), activation='relu') %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation='relu') %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=128, kernel_size=c(3,3), activation='relu') %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters=256, kernel_size=c(3,3), activation='relu') %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_dropout(rate=0.25) %>%
  layer_flatten() %>%
  layer_dense(units=128, activation='relu') %>%
  layer_dropout(rate=0.5) %>%
  layer_dense(units=2, activation='softmax')

summary(model)
# 
 model %>% compile(loss=loss_categorical_crossentropy,
                   optimizer=optimizer_adadelta(), metrics=c('accuracy'))
# 
# 
# # 
# make_train <- tensor_slices_dataset(
#   list(tf$cast(train_array, tf$float32), 
#        tf$one_hot(train_data$class_id, 30L)))
# make_test <- list(tf$cast(test_array, tf$float32), tf$one_hot(train_data$class_id, 30L))
# 
# make_input <- function(ds, batch_size){
#   ds <- ds %>% dataset_batch(batch_size)
# }
# 
# train_gen <- make_input(make_train, 32)
# test_gen <- make_input(make_test, 32)
# 
# model %>% fit_generator(generator=train_gen, 
#                         steps_per_epoch = nrow(train_array)/32,
#                         epochs = 10,
#                         validation_data = test_gen,
#                         validation_steps = nrow(test_array)/32,
#                         batch_size = 32)
                        

y_test <- tf$one_hot(train_data$class_id, 30L)
history <- model %>% fit(x=train_array, y=train_data$class_id,
                         epochs=30, batch_size=32, validation_split=0.2)
