training_config = dict(
    epoch = 10,
    batch_size = 256,
    num_workers = 4,
    log_iter = 1
)

optim_config = dict(
    optim = "adam",
    lr = 3e-4
)

validation_config = dict(
    early_stopping_epoch    = 50,
    test_accuracy_log_epoch = 1
)

model_config = dict(
    layer_type     = "GCN",
    embedding_size = 512,
    heads          = 4,
    dropout        = 0.6,
)

data_config = dict(
    evaluate        = False,
    num_classes     = 2,
    db_root         = "/users/Etu6/28718016/Data/BinaryClassification/train",
    test_root       = "/users/Etu6/28718016/Data/BinaryClassification/val",
    validation_root = "/users/Etu6/28718016/Data/BinaryClassification/val",
    normalize       = dict(l1 = False, l2= False)
)

config = dict(training_config   = training_config,
              optim_config      = optim_config,
              validation_config = validation_config,
              model_config      = model_config,
              data_config       = data_config)