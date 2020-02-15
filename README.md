# PuzzleCounter
R script to count # of pieces remaining in a jigsaw puzzle

## Overview: 
I'm curious about the pieces place vs. time graph for jigsaw puzzles. I suspect that there is a characteristic shape, or set of shapes, to the curve that are similar between puzzles. For example: no pieces placed while edge pieces are being found, then quick placement while the edge is built, then slower for a while, then very fast at the end. To study that, I want to build an app to record the timestamp whenever a piece is placed and plot pieces remaining vs. time.

## Build sequence
Step 1: Voice recognition model. I don't want to have to interact physically with the computer while I'm working on a puzzle, so I need a voice interface for alerting the script each time a piece is placed.

 - General data prep / model training flow based on 'MNIST Example' on the Keras for R blog and 'Convolutional Neural Networks in R' from poissonisfish

  https://blog.rstudio.com/2017/09/05/keras-for-r/

  https://poissonisfish.com/2018/07/08/convolutional-neural-networks-in-r/

 - Spectrogram creation roughly follows Hansen Johnson's 'Spectrograms in R' post with further insight from 'Audio Background' post on Rstudio blog.

  https://hansenjohnson.org/post/spectrograms-in-r/

  https://blogs.rstudio.com/tensorflow/posts/2019-02-07-audio-background/

 - The voice recognition model architecture, training data, and alluvial plot are borrowed from 'Simple Audio Classification with Keras' post from TensorFlow for R Blog. https://blogs.rstudio.com/tensorflow/posts/2018-06-06-simple-audio-classification-keras/

 - Audio commands datafile is not included in repository due to size. It can be downloaded from: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz and should be untarred into data/speech_commands_v0.01 in the project folder.
