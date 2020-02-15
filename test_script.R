# Scratchpad script for working out bugs and logic

## open wav and make spectrogram
# following https://hansenjohnson.org/post/spectrograms-in-r/
library(tuneR)
library(signal)
library(oce)
filename <- 'data/speech_commands_v0.01/bird/3e7124ba_nohash_0.wav'

n = 64727
k = 1
evens = seq(k, 16000, k)

# read in audio file
data <- readWave(filename)

# extract signal
snd <- data@left
#snd <- snd[evens]

# determine duration
dur = length(snd)/data@samp.rate*k

# determine sample rate
fs <- data@samp.rate/k

# plot waveform
plot(snd, type='l', xlab='Samples', ylab='Amplitude')

# window size in points
window <- 480

# number of points to use for the fft
nfft = as.integer(2^(trunc(log(window, 2))+1))

# window overlap in points
overlap <-320

# create spectrogram
spec <- specgram(x=snd, n=nfft, Fs=fs, window=window,
                 overlap=overlap)


P <- log10(abs(spec$S)+1e-6)
image(x=spec$t, y=spec$f, z=t(P),
      ylab='Frequency[Hz]', xlab='Time[s]',
      main=paste(df$class[n], " ", window))

spec_vec <- as.vector(t(P))
remat <- matrix(spec_vec, ncol = nrow(P))
#image(x=spec$t, y=spec$f, z=remat,
#      ylab='Frequency[Hz]', xlab='Time[s]',
#      main=paste(df$class[n], " ", k))

## Open wav and make spectrogram by tensorflow method
# Following https://blogs.rstudio.com/tensorflow/posts/2019-02-07-audio-background/

# library(keras)
# use_implementation('tensorflow')
# 
# library(tensorflow)
# #tfe_enable_eager_execution(device_policy='silent')
# 
# fname <- 'data/speech_commands_v0.01/bird/00b01445_nohash_0.wav'
# fname <- 'data/speech_commands_v0.01/bird/1746d7b6_nohash_0.wav'
# #fname <- df$fname[n]
# wav <- tf$audio$decode_wav(tf$io$read_file(fname))
# 
# sampling_rate <- wav$sample_rate %>% as.numeric()
# sampling_rate
# 
# samples <- wav$audio
# samples <- samples %>% tf$transpose(perm=c(1L, 0L))
# samples
# 
# window_size_ms <- 30
# window_stride_ms <- 10
# 
# samples_per_window <- as.integer(sampling_rate*window_size_ms/1000)
# stride_samples <- as.integer(sampling_rate*window_stride_ms/1000)
# # n_periods <- length(seq(samples_per_window/2, 
# #                         sampling_rate-samples_per_window/2,
# #                         stride_samples))
# n_periods <- 
#   tf$shape(
#     tf$range(samples_per_window%/%2L, 
#              16000L-samples_per_window%/%2L, 
#              stride_samples%/%1L))[1]+1L
# fft_size <- as.integer(2^trunc(log(samples_per_window, 2))+1)
# 
# stft_out <- tf$signal$stft(
#   samples,
#   frame_length=samples_per_window,
#   frame_step=stride_samples)
# stft_out
# 
# magnitude_spectrograms <- tf$abs(stft_out)
# magnitude_spectrograms
# 
# log_magnitude_spectrograms <- tf$math$log(magnitude_spectrograms + 1e-6, 10)
# log_magnitude_spectrograms
# 
# lms_matrix <- as.matrix(log_magnitude_spectrograms[1,,])
# image(z=lms_matrix)


# #Make an alluvial plot with ggplot
# library(ggplot2)
# library(ggalluvial)
# library(dplyr)
# 
# dat <- as.data.frame(UCBAdmissions)
# head(dat, n=12)
# head(UCBAdmissions, n=20)
# 
# #keep_real <- test_y
# #keep_pred <- model %>% predict_classes(test_specs)
# 
# results = table(Actual = keep_real[,2], Pred = keep_pred)
# results = as.data.frame(results)
# results$Correct = results$Actual == results$Pred
# ggplot(results, aes(y = Freq, axis1 = Pred, axis2 = Actual)) +
#   geom_alluvium(aes(fill=Correct), width = 1/12) +
#   geom_stratum(width = 1/12, fill = 'black', color = 'grey') +
#   geom_label(stat = 'stratum', infer.label = TRUE) +
#   scale_x_discrete(limits = c('Predicted', 'Actual'), 
#                    expand = c(0.05, 0.05)) +
#   scale_fill_brewer(type = 'qual', palette = 'Set1') +
#   ggtitle("CNN Voice Recognition")
# results
