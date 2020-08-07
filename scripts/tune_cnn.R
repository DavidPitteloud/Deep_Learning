# Load packages
library("here")
library("keras")
library("reticulate")
library("tensorflow")
library("tfruns")
library(tidyverse)


#training_run

tuning_run(
  file = here::here("scripts/train_cnn.R"),
  flags = list(L1 = c(0.001),
               L2 = c(0.002),
               dropout1 = c(0.3,0.4),
               dropout2 = c(0.1),
               dropout3 = c(0.3,0.4),
               filter1 = c(96,196),
               filter2 = c(96,196)
  )
)









