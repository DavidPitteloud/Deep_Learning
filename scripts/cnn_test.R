# Load packages
library("keras")
library("tfruns")
library("tidyverse")

# 0. Define hyperparameter flags
#-------------------------------------------------------------------------------

FLAGS <- flags(
  flag_numeric("L1", 0.001),
  flag_numeric("L2", 0.002),
  flag_numeric("dropout1", 0.3),
  flag_numeric("dropout2", 0.1),
  flag_numeric("dropout3", 0.3),
  flag_numeric("filter1", 96),
  flag_numeric("filter2", 96)
)


# 0. Define the generator variable
#-------------------------------------------------------------------------------

generator <- image_data_generator(rescale = 1/255,
                                  validation_split = 0.2)

training_generator <- flow_images_from_directory(
  directory = here::here("data/plot/final_plot/train/"),
  generator = generator,
  target_size = c(28, 28),
  batch_size = 32,
  subset = "training"
)

validation_generator <- flow_images_from_directory(
  directory = here::here("data/plot/final_plot/train/"),
  generator = generator,
  target_size = c(28, 28),
  batch_size = 32,
  subset = "validation"
)

# 1. Define the model
#-------------------------------------------------------------------------------

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, 
                kernel_size = c(2,2),
                padding = 'Same',
                input_shape = c(28,28,3), 
                activation = "relu",
                kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>%
  layer_batch_normalization() %>% 
  layer_conv_2d(filters = 32, 
                kernel_size = c(2,2),
                padding = 'Same', activation = "relu",
                kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>% 
  layer_batch_normalization() %>% 
  layer_average_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = FLAGS$dropout1) %>% 
  
  layer_conv_2d(filters = FLAGS$filter1, 
                kernel_size = c(2,2),
                padding = 'Same',
                activation = "relu", 
                kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>%
  layer_batch_normalization() %>% 
  layer_conv_2d(filters = FLAGS$filter1, 
                kernel_size = c(2,2),
                padding = 'Same', 
                activation = "relu",
                kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>% 
  layer_batch_normalization() %>% 
  layer_average_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = FLAGS$dropout1) %>% 
  
  layer_conv_2d(filters = FLAGS$filter2, 
                kernel_size = c(2,2),
                padding = 'Same', 
                activation = "relu",
                kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>%
  layer_batch_normalization() %>% 
  layer_conv_2d(filters = FLAGS$filter2,
                kernel_size = c(2,2),
                padding = 'Same', 
                activation = "relu",
                kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>%
  layer_batch_normalization() %>% 
  layer_average_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = FLAGS$dropout1) %>% 
  
  layer_flatten() %>% 
  layer_dense(units = 384, 
              activation = "relu",
              kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>% 
  layer_dropout(rate = FLAGS$dropout2) %>% 
  layer_dense(units = 192, 
              activation = "relu",
              kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>% 
  layer_dropout(rate = FLAGS$dropout3) %>% 
  layer_dense(units = 3, activation = 'softmax')

# 2. Compile the model
#-------------------------------------------------------------------------------

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001),
  metric = "accuracy"
)

# 3. Fit the model
#-------------------------------------------------------------------------------

model %>% fit_generator(
  generator = training_generator,
  steps_per_epoch = training_generator$n / training_generator$batch_size,
  epochs = 50,
  validation_data = validation_generator,
  validation_steps = validation_generator$n / validation_generator$batch_size,
  callbacks = list(callback_early_stopping(patience = 20,
                                           monitor = "val_loss",
                                           restore_best_weights = TRUE),
                   callback_reduce_lr_on_plateau(patience = 10, factor = 0.15))
)


model %>% save_model_hdf5(here::here(("results/model_graph_final.hdf5")))