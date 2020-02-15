# Following basic tutorial for Keras in R from:
# https://keras.rstudio.com/

library(keras)
library(pbapply)
library(tibble)
library(stringr)
library(tuneR)
library(signal)
library(ggplot2)
library(ggalluvial)
library(alluvial)
library(dplyr)

rm(list=ls())
gc()

# # Get mnist digit data
# mnist <- dataset_mnist()
# x_train <- mnist$train$x
# y_train <- mnist$train$y
# x_test <- mnist$test$x
# y_test <- mnist$test$y

# Get speech file list
files <- fs:: dir_ls(path = 'data/speech_commands_v0.01/',
                     recurse = TRUE, glob = '*.wav')

# keep all files except those with background_noise in the name
files <- files[!str_detect(files, "background_noise")]

# Make a tibble of filenames, class (word spoken), and 
# class as an integer
df <- tibble(fname = files,
             class = fname %>% str_extract("1/.*/") %>%
               str_replace_all("1/", "") %>%
               str_replace_all("/", ""),
             class_id = class %>% as.factor() %>% as.integer() - 1L)

# Check data
check = sample(nrow(df), 20)
df[check,]

# Make training and test subsets
size <- 64720
split <- 0.8
#options <- which(df$class_id < 31)
options <- sample(which(df$class_id < 31), size)
train <- options[1:as.integer(split*size)]
test <- options[(as.integer(split*size)+1):size]

# Define spectrogram creation function
make_spec <- function(filename, window, overlap){
  data <- readWave(filename) #read in audio
  snd <- data@left # extract signal
  if(length(snd) < 16000) { # pad length if < 1 sec
    snd <- c(snd, rep(mean(snd), 16000))[1:16000]
  }
  
  # Create spectrogram
  spec <- specgram(x=snd, n=window, Fs = 16000, overlap=overlap)
  P <- abs(spec$S)
  P <- P/max(P)
  P <- (log10(P + 1e-6) + 6) / 6
  check_nan = which(is.na(P))
  if (length(check_nan !=0)) {
    print(paste("NaN in ", filename))
  }
  return(P)
}

# Define data generation funtion
data_gen <- function(df, window, overlap){
  print(paste('Start processing', nrow(df), 'images'))
  #print(length(df$fname))
  specs <- pbsapply(df$fname, FUN=make_spec, 
                    window=window, overlap=overlap, 
                    simplify = 'array')
  #print(dim(specs))
  #specs <- aperm(specs, c(3, 2, 1)) # 3 dimensional array
  dim(specs) <- c(dim(specs), 1)
  specs <- aperm(specs, c(3, 2, 1, 4))
  print('Finished! Check plot.')
  image(z=specs[1,,,], main=df$fname[1])
  return(specs)
}

# Set up spectrogram creation
# Note: wav files need sample rate 16000 Hz and length <= 1 sec

window <- 480
stride <- 240
overlap <- window - stride
n_times <- length(seq(window/2, 16000-window/2, stride))

train_specs <- data_gen(df[train,], window, overlap)
gc()

# Refactor the train and test sets together then make one-hot array
newfacs <- as.numeric(as.factor(c(df$class[train], df$class[test])))-1
types = length(unique(newfacs))
train_y <- to_categorical(newfacs[1:as.integer(size*split)], types)
test_y <- to_categorical(newfacs[(as.integer(size*split)+1):size], types)
#train_y
#test_y

#

# # Reshape data into matrices by flattening images and normalize
# #x_train <- array_reshape(x_train, c(nrow(x_train), 784))
# #x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# x_train <- x_train / 255
# x_test <- x_test / 255
# 
# # Encode y's as one-hot categorical matrices
# y_train <- to_categorical(y_train, 10)
# y_test <- to_categorical(y_test, 10)
# 
# Define model
model <- keras_model_sequential()
# model %>%
#   layer_flatten(input_shape = c(65, 240)) %>%
#   layer_dense(units = 256, activation = 'relu') %>%
#   layer_dropout(rate = 0.4) %>%
#   layer_dense(units = 128, activation = 'relu') %>%
#   layer_dropout(rate = 0.3) %>%
#   layer_dense(units = types, activation = 'softmax')
model %>%
  layer_conv_2d(input_shape=c(65, 240, 1), filters=32,
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
  layer_dense(units=types, activation='softmax')
summary(model)

# Compile model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(train_specs, train_y, epochs = 30,
                         batch_size = 128, validation_split = 0.2)
plot(history)

test_specs <- data_gen(df[test,], window, overlap)
model %>% evaluate(test_specs, test_y)
classes <- model %>% predict_classes(test_specs) 
probs <- model %>% predict_proba(test_specs)
# mnist$test$y

df[test,]
freq_table <- df[test,] %>% 
  mutate(pred_class_id = as.integer(classes))
freq_table
mixdf <- df[test,] %>% distinct(class_id, class)
mixdf
mixdf <- mixdf %>% rename(pred_class = class)
mixdf
freq_table <- left_join(freq_table, mixdf, by = c('pred_class_id' = 'class_id'))
freq_table
freq_table <- freq_table %>% mutate(correct = pred_class == class) %>% 
  count(pred_class, class, correct)
freq_table

alluvial(
  freq_table %>% select(class, pred_class), 
  freq = freq_table$n,
  col = ifelse(freq_table$correct, "lightblue", 'red'),
  border = ifelse(freq_table$correct, 'lightblue', 'red'),
  alpha = 0.6, hide = freq_table$n < 9
)

model %>% save_model_hdf5("speech_model_full_dataset_train01.h5")
save(train, test, df, 
     file = "trainer_inputs_full_dataset_train01.RData")
