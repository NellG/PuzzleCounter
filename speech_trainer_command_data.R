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
ntrain <- 300
ntest <- 300

all <- seq(nrow(df))
train <- sample(all, ntrain, replace=FALSE)
test <- sample(all[-train], ntest, replace=FALSE)

# set up spectrogram creation
window_size_ms <- 30 # window in ms
window_stride_ms <- 10 # step size between windows in ms
# note: files should have 16000 samples per second
window <- as.integer(16*window_size_ms) # samples per window
stride <- as.integer(16*window_stride_ms) # samples per stride
n_freq <- as.integer(2^(trunc(log(window, 2)))+1)
n_chunks <- length(seq(window_size/2, 16000-window_size/2, stride))

# tensorflow data generator function
data_generator <- function(df, window_size_ms, window_stride_ms){
  #print('1')
  dat <- tf$io$read_file(tf$reshape(df$fname[[1]], list()))
  #print('2')
  sampling_rate <- tf$audio$decode_wav(dat)$sample_rate
  
  window <- (sampling_rate*window_size_ms)%/%1000L # samples per window
  stride <- (sampling_rate*window_stride_ms)%/%1000L # samples per stride
  n_freq <- (2^tf$math$ceil(tf$math$log(tf$cast(window, tf$float32))/
                              tf$math$log(2))/2+1L) %>%
    tf$cast(tf$int32)
  n_chunks <- tf$shape(tf$range(window %/% 2L, 16000L-window%/%2L, stride))[1]+1L
  print('3')
  
  ds <- tensor_slices_dataset(df) %>% dataset_shuffle(buffer_size=50)
  print(ds)

  
  ds <- ds %>% dataset_map(function(obs){
    print('4')
    audio_binary <- tf$io$read_file(tf$reshape(obs$fname, shape=list()))
    wav <- tf$audio$decode_wav(audio_binary, desired_channels=1)
    samples <- wav$audio
    samples <- samples %>% tf$transpose(perm=c(1L, 0L))
    print(samples)
    print('5')
    stft_out <- tf$signal$stft(samples, frame_length=window, frame_step=stride)
    mag_spec <- tf$abs(stft_out)
    log_mag_spec <- tf$math$log(mag_spec+1e-6)
    print('6')
    response <- tf$one_hot(obs$class_id, 30L)
    input <- tf$transpose(log_mag_spec, perm=c(1L, 2L, 0L))
    print('7')
    list(input, response)
  }) 
  
  ds <- ds %>% dataset_repeat()
  print('8')
  print(ds)
  
  ds <- ds %>% dataset_batch(batch_size=32)
  print('9')
  ds
}


ds_train <- data_generator(df[train,], window_size_ms=30L, window_stride_ms=10L)
print('10')

sess <- tf$Session()
batch <- next_batch(ds_train)
str(sess$run(batch))

# make_spec <- function(filename, window, overlap){
#   data <- readWave(filename) # read in audio
#   snd <- data@left # extract signal
#   n <- as.integer(2^(trunc(log(window, 2)+1)))
#   if (length(snd) < 16000) {
#         snd <- c(snd, rep(0, 16000))[1:16000]
#   }
# 
#   # create spectrogram and crop at max frequency
#   spec <- specgram(x=snd, n=n, Fs=data@samp.rate, 
#                    window=window, overlap=overlap)
#   P <- abs(spec$S)
#   P <- log10(P+0.01) # normalize
#   #print(paste('Row: ', nrow(P), ' Col: ', ncol(P), ' n: ', n))
#   #print(paste('Max freq:', max(spec$f), ' Times: ', spec$t))
#   spec_vec <- as.vector(t(P))
#   # to turn the spec lists back into matrices use matrix(spec, ncol = n/2)
#   return(spec_vec)
# }
# 
# data_generator <- function(df, window, overlap){
#   
#   # set up parameters for spectrogram creation
#   # window and overlap are in samples
# 
#   print(paste('Start processing', nrow(df), 'images'))
#   df$spec <- pblapply(df$fname, make_spec, window=window, overlap=overlap)
#   
#   print('Success!')
#   return(df)
#   
# }
# 
# train_data <- data_generator(df[train,], window, overlap)
# test_data <- data_generator(df[test,], window, overlap)
# 
# 
# test_img <- t(matrix(as.numeric(train_data$spec[[1]]), ncol=n_freq/2))
# image(z=t(test_img), axes=F)
# 
# # Build training  and test arrays
# train_array <- unlist(t(train_data$spec))
# dim(train_array) <- c(n_time, n_freq/2, length(train_data$spec), 1)
# 
# test_array <- unlist(t(test_data$spec))
# dim(test_array) <- c(n_time, n_freq/2, length(test_data$spec), 1)
# 
# # Reorder dimensions
# train_array <- aperm(train_array, c(3, 1, 2, 4))
# test_array <- aperm(test_array, c(3, 1, 2, 4))
# 
# image(z=(test_array[4,,,1]))
# 
# # Build convolutional neural network model
# 
# model <- keras_model_sequential()
# 
# # model specified by HJ post
# # model %>% 
# #   layer_conv_2d(kernel_size=c(3,3), filter=32, activation='relu',
# #                 padding='same', input_shape=c(n_time, n_freq/2, 1),
# #                 data_format='channels_last') %>%
# #   layer_conv_2d(kernel_size=c(3,3), filter=32, activation='relu',
# #                 padding='valid') %>%
# #   layer_max_pooling_2d(pool_size=2) %>%
# #   layer_dropout(rate=0.25) %>%
# #   
# #   layer_conv_2d(kernel_size=c(3,3), filter=64, strides=2,
# #                 activation='relu', padding='same') %>%
# #   layer_conv_2d(kernel_size=c(3,3), filter=64, activation='relu',
# #                 padding='valid') %>%
# #   layer_max_pooling_2d(pool_size=2) %>%
# #   layer_dropout(rate=0.25) %>%
# #   
# #   layer_flatten() %>%
# #   layer_dense(units=50, activation='relu') %>%
# #   layer_dropout(rate=0.25) %>%
# #   layer_dense(units=1, activation='sigmoid')
# 
# # model specified by tensorflow post
# model %>% 
#   layer_conv_2d(input_shape=c(n_time, n_freq/2,1), filters=32,
#                         kernel_size=c(3,3), activation='relu') %>%
#   layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   layer_conv_2d(filters=64, kernel_size=c(3,3), activation='relu') %>%
#   layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   layer_conv_2d(filters=128, kernel_size=c(3,3), activation='relu') %>%
#   layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   layer_conv_2d(filters=256, kernel_size=c(3,3), activation='relu') %>%
#   layer_max_pooling_2d(pool_size=c(2,2)) %>%
#   layer_dropout(rate=0.25) %>%
#   layer_flatten() %>%
#   layer_dense(units=128, activation='relu') %>%
#   layer_dropout(rate=0.5) %>%
#   layer_dense(units=30, activation='softmax')
# 
# summary(model)
# 
# model %>% compile(loss=loss_categorical_crossentropy,
#                   optimizer=optimizer_adadelta(), metrics=c('accuracy'))
# 
# 
# 
# model %>% fit_generator(generator=)
# 
# # history <- model %>% fit(x=train_array, y=train_data$class_id,
# #                          epochs=5, batch_size=100, validation_split=0.2)
