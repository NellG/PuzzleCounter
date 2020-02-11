## open wav and make spectrogram
# following https://hansenjohnson.org/post/spectrograms-in-r/
library(tuneR)
library(signal)
library(oce)
filename <- 'data/speech_commands_v0.01/bed/00176480_nohash_0.wav'

n = 3

# read in audio file
data <- readWave(df$fname[n])

# extract signal
snd <- data@left

# determine duration
dur = length(snd)/data@samp.rate

# determine sample rate
fs <- data@samp.rate

# plot waveform
plot(snd, type='l', xlab='Samples', ylab='Amplitude')

# number of points to use for the fft
nfft = 16000

# window size in points
window <- 480

# window overlap in points
overlap <-120

# create spectrogram
spec <- specgram(x=snd, n=nfft, Fs=fs, window=window, 
                 overlap=overlap)


P = abs(spec$S)
P <- P/max(P)
P <- 10*log10(P)
image(x=spec$t, y=spec$f, z=t(P), 
      ylab='Frequency[Hz]', xlab='Time[s]', main=df$class[n])
