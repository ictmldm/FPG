import yaml
import torch

# Read the configuration file
with open('./config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set paths
paths = config['paths']
news_file_path_sp = paths['news_file_path_sp']
news_file_path = paths['news_file_path']
train_file_path = paths['train_file_path']
dev_file_path = paths['dev_file_path']
test_file_path = paths['test_file_path']
processed_data_path = paths['processed_data_path']
model_path = paths['model_path']
fact_ckpt = paths['fact_ckpt']

# Set model parameters
model_config = config['model']
bart_name = model_config['bart_name']
model_type = model_config['model_type']
user_type = model_config['user_type']
device = torch.device(model_config['device'])
encoder_cross_attention_heads = model_config['encoder_cross_attention_heads']

# Set training parameters
training_config = config['training']
limit = training_config['limit']
max_news_title_length = training_config['max_news_title_length']
max_news_body_length = training_config['max_news_body_length']
max_click_length = training_config['max_click_length']
batch_size = training_config['batch_size']
pretrain_lr = float(training_config['pretrain_lr'])
step_2_lr = float(training_config['step_2_lr'])
step_3_lr = float(training_config['step_3_lr'])
step_4_lr = float(training_config['step_4_lr'])

pretrain_epoch_num = training_config['pretrain_epoch_num']
train_step_2_epoch_num = training_config['train_step_2_epoch_num']
train_step_3_epoch_num = training_config['train_step_3_epoch_num']
train_step_4_epoch_num = training_config['train_step_4_epoch_num']

beam_size = training_config['beam_size']

# Set training steps
steps = config['steps']
TRAIN_STEP_1 = steps['TRAIN_STEP_1']
TRAIN_STEP_2 = steps['TRAIN_STEP_2']
TRAIN_STEP_3 = steps['TRAIN_STEP_3']
TRAIN_STEP_4 = steps['TRAIN_STEP_4']

# Print all parameters
print("Paths:")
print(f"  News file path (simplified): {news_file_path_sp}")
print(f"  News file path: {news_file_path}")
print(f"  Train file path: {train_file_path}")
print(f"  Dev file path: {dev_file_path}")
print(f"  Test file path: {test_file_path}")
print(f"  Processed data path: {processed_data_path}")
print(f"  Model path: {model_path}")
print(f"  Fact checkpoint: {fact_ckpt}")

print("\nModel Configuration:")
print(f"  BART name: {bart_name}")
print(f"  Model type: {model_type}")
print(f"  User type: {user_type}")
print(f"  Device: {device}")
print(f"  Encoder cross attention heads: {encoder_cross_attention_heads}")

print("\nTraining Configuration:")
print(f"  Limit: {limit}")
print(f"  Max news title length: {max_news_title_length}")
print(f"  Max news body length: {max_news_body_length}")
print(f"  Max click length: {max_click_length}")
print(f"  Batch size: {batch_size}")
print(f"  Pretrain learning rate: {pretrain_lr}")
print(f"  Step 2 learning rate: {step_2_lr}")
print(f"  Step 3 learning rate: {step_3_lr}")
print(f"  Step 4 learning rate: {step_4_lr}")
print(f"  Pretrain epoch num: {pretrain_epoch_num}")
print(f"  Train step 2 epoch num: {train_step_2_epoch_num}")
print(f"  Train step 3 epoch num: {train_step_3_epoch_num}")
print(f"  Train step 4 epoch num: {train_step_4_epoch_num}")
print(f"  Beam size: {beam_size}")

print("\nTraining Steps:")
print(f"  TRAIN_STEP_1: {TRAIN_STEP_1}")
print(f"  TRAIN_STEP_2: {TRAIN_STEP_2}")
print(f"  TRAIN_STEP_3: {TRAIN_STEP_3}")
print(f"  TRAIN_STEP_4: {TRAIN_STEP_4}")
