from addict import Dict

__all__ = ['model', 'solver']

# Configuration for setting up a test model
model = Dict()

model.mode = 'train'
model.batch_size = 128
model.num_samples = None  # filled while calling
model.time_steps = 26
model.vocab_size = 9000
model.embedding_size = 512
model.lstm_cells = 512
model.dropout = 0.7
model.img_input_feed = 'image_feature'
model.cap_input_feed = 'input_feed'
model.resume_from_model_path = None
model.model_path = None  # used in case of 'eval' or 'inference'

# Configuration for training the test model
solver = Dict()

solver.num_epochs = 10
solver.save_model_dir = None
solver.max_to_keep = 2
solver.ckpt_epoch_freq = 2
solver.train_clip_gradients = 5.0

# fixed learning rate with Adam
solver.optimizer = 'Adam'
solver.learning_rate = 0.001

# decay learning rate with SGD
# solver.optimizer = 'SGD'
# solver.learning_rate = 2.0
# solver.lr_decay_method = 'piecewise'
# solver.lr_decay_boundaries = "2,5,7"
# solver.lr_decay_values = "2.0,1.0,0.5,0.1"

