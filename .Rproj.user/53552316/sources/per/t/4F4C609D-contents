# Following along with the Tensorflow Blog voice 
# recognition post from here:
# https://blogs.rstudio.com/tensorflow/posts/2018-06-06-simple-audio-classification-keras/

library(stringr)
library(dplyr)
library(tfdatasets)

# from fs (file system operations) package borrow dir_ls
files <- fs::dir_ls(
  path = "data/speech_commands_v0.01/",
  recursive = TRUE,
  glob = "*.wav"
)

# keep all files except those with background_noise in the name
files <- files[!str_detect(files, "background_noise")]

df <- tibble(
  fname = files,
  class = fname %>% str_extract("1/.*/") %>%
    str_replace_all("1/", "") %>%
    str_replace_all("/", ""),
  class_id = class %>% as.factor() %>% as.integer() - 1L
)
