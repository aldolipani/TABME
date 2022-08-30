config = {
    # Model
    'input_img_size': (512, 512),
    'max_seq_length': 150,

    # Training
    'num_epochs': 30,
    'batch_size': 64,
    'patience': 5,
    'num_batch_eval': 9999999, # very high number -> evaluate at the end of the epoch
}