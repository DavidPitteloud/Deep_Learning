  # Load packages
  library("keras")
  library("tfruns")
  library("tidyverse")
  
  # 0. Define hyperparameter flags
  #-------------------------------------------------------------------------------
  
  FLAGS <- flags(
    flag_numeric("L1", 0.001),
    flag_numeric("L2", 0.001),
    flag_numeric("filter1", 64),
    flag_numeric("filter2", 64),
    flag_numeric("dropout1", 0.1),
    flag_numeric("kernel_size1", 2),
    flag_numeric("kernel_size2", 2)
  )
  

  # 1. Define the model
  #-------------------------------------------------------------------------------
  
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = FLAGS$filter1, 
                  kernel_size = c(FLAGS$kernel_size1,1), 
                  padding = 'Same',
                  input_shape = c(15,15,1),
                  activation = "relu",
                  kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>%
    layer_conv_2d(filters = FLAGS$filter2, 
                  kernel_size = c(FLAGS$kernel_size2,1),
                  padding = 'Same', 
                  activation = "relu",
                  kernel_regularizer = regularizer_l1_l2(l1 = FLAGS$L1, l2 = FLAGS$L2)) %>% 
    layer_dropout(rate = FLAGS$dropout1) %>% 
    layer_flatten() %>% 
    layer_dense(units = 1024, activation = "relu") %>% 
    layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 256, activation = "relu") %>% 
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 3, activation = 'softmax')
  
  # 2. Compile the model
  #-------------------------------------------------------------------------------
  
  model %>% compile(
    optimizer = optimizer_rmsprop(lr = 0.0001),
    loss = "sparse_categorical_crossentropy",
    metric = "accuracy"
  )
  
  # 3. Fit the model
  #-------------------------------------------------------------------------------
  
  model %>% fit(
    x = train_x,
    y = train_y, 
    batch_size = 128,
    epochs = 20,
    shuffle = FALSE,
    callbacks = list(callback_early_stopping(patience = 20,
                                             monitor = "val_loss",
                                             restore_best_weights = TRUE),
                     callback_reduce_lr_on_plateau(patience = 5, factor = 0.1)),
    validation_split = 0.2)
  
  
  


