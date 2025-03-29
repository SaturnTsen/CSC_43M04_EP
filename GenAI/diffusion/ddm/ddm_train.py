from tqdm import tqdm  # tqdm 用于进度条    

#%%

import os
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader

from .net import (DiffusionModel, UNet,
    linear_diffusion_schedule,
    cosine_diffusion_schedule,
    offset_cosine_diffusion_schedule)
from utils import ImageGenerator

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a diffusion model.")

    parser.add_argument('--dataset_path', type=str, default=Path(__file__) / '..' / '..' / '..' / '..' / 'data' / 'flower-dataset',)
    parser.add_argument('--dataset_repetitions', type=int, default=5, help='Number of times to repeat the dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    
    parser.add_argument('--image_size', type=int, default=64, help='Size of the images')
    parser.add_argument('--noise_embedding_size', type=int, default=32, help='Size of the noise embedding')
    parser.add_argument('--ema', type=float, default=0.999, help='Exponential moving average decay')

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for the optimizer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    
    parser.add_argument('--checkpoint_path', type=str, default=Path(__file__) / '..' / '..' / 'checkpoint' / 'checkpoint.ckpt', help='Path to save the checkpoint')
    parser.add_argument('--load_model', type=bool, default=True, help='Load existing checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Load existing checkpoint')

    args = parser.parse_args()

    DATASET_REPETITIONS = args.dataset_repetitions
    BATCH_SIZE = args.batch_size
    
    NOISE_EMBEDDING_SIZE = args.noise_embedding_size
    IMAGE_SIZE = args.image_size
    EMA = args.ema
    
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    EPOCHS = args.epochs
    
    LOAD_MODEL = args.load_model
    checkpoint_path = args.checkpoint_path
    device = args.device
    
    # Prepare training data
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # load dataset (already normalized to [0,1])
    train_dataset = Flowers102(
        root = args.dataset_path,
        split="train",
        transform=transform,
        download=True
    )

    # Repeat the dataset by concatenating it DATASET_REPETITIONS times
    train_dataset_repeated = torch.utils.data.ConcatDataset([train_dataset] * DATASET_REPETITIONS)

    train_data = DataLoader(
        train_dataset_repeated, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    ddm = DiffusionModel(UNet, offset_cosine_diffusion_schedule, EMA=0.999).to(device)
    
    # Checkpoint
    if os.path.exists(checkpoint_path):
        if not LOAD_MODEL:
            raise FileExistsError(f"Checkpoint already exists at {checkpoint_path}")
        else:
            print('Loading checkpoint from', checkpoint_path)
            
        ddm.load_state_dict(torch.load(checkpoint_path))
        
    # Initialize SummaryWriter
    writer = SummaryWriter()

    # Train the model
    optimizer = torch.optim.AdamW(ddm.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.L1Loss()
    image_generator_callback = ImageGenerator(ddm, num_img=10)

    global_step = 0  # 用于 TensorBoard 每个 batch 的 step 计数 
    for epoch in range(EPOCHS):
        epoch_bar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, _ in epoch_bar:
            images = images.to(next(ddm.parameters()).device)   

            # 执行一次训练 step
            loss = ddm.train_step(images, optimizer, criterion) 

            # 更新 tqdm 上显示的信息
            epoch_bar.set_postfix(loss=loss)    

            # 写入 TensorBoard 每 batch 的 loss
            writer.add_scalar('Loss/train_step', loss, global_step)
            global_step += 1    

        # 每个 epoch 结束后生成图像
        image_generator_callback.on_epoch_end(epoch)    

        # 保存模型
        torch.save(ddm.state_dict(), checkpoint_path)    

        # 写入 TensorBoard 每个 epoch 的 loss
        writer.add_scalar('Loss/train_epoch_avg', loss, epoch)
