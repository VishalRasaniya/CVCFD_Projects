#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import time
import logging
import sys

# --------------------------------------------------
# Configuration Parameters
# --------------------------------------------------
# --- Paths ---
BASE_OUTPUT_DIR = "./training_run_{timestamp}"
LOGS_DIR_NAME = "logs"
CHECKPOINTS_DIR_NAME = "checkpoints"
PLOTS_DIR_NAME = "plots"
DATA_PATH = "/home/vishal.rasaniya/mlcfd_project/Temperature_contours_CNN/"  # Adjust this path

# --- Model & Image ---
IMG_HEIGHT = 64
IMG_WIDTH = 64
NUM_CHANNELS = 3  # RGB

# --- Sequence ---
TIME_STEPS_IN = 10
TIME_STEPS_OUT = 50

# --- Training ---
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
TEACHER_FORCING_RATIO = 0.5
GRADIENT_CLIPPING_VALUE = 1.0
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# --- Architecture ---
HIDDEN_DIM = 64
ENCODER_LAYERS = 2
DECODER_LAYERS = 2
KERNEL_SIZE = 3

# --- Hardware & Performance ---
NUM_WORKERS = 4
PIN_MEMORY = True
SAVE_CHECKPOINT_EPOCH_INTERVAL = 10

# --------------------------------------------------
# Global Variables
# --------------------------------------------------
logger = None
device = None

# --------------------------------------------------
# Logging Setup Function
# --------------------------------------------------
def setup_logging(log_dir, run_timestamp):
    """Configures the logging module."""
    global logger
    log_filename = os.path.join(log_dir, f"training_{run_timestamp}.log")
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("ConvLSTM_Training")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    try:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logger at {log_filename}: {e}")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(console_handler)

    logger.info("Logging setup complete.")
    logger.info(f"Log file: {log_filename}")
    return logger

# --------------------------------------------------
# Model Definitions
# --------------------------------------------------

# 1) ConvLSTMCell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)
        self.bias = bias
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        chunk_size = self.hidden_dim
        i = torch.sigmoid(gates[:, 0*chunk_size:1*chunk_size, :, :])
        f = torch.sigmoid(gates[:, 1*chunk_size:2*chunk_size, :, :])
        o = torch.sigmoid(gates[:, 2*chunk_size:3*chunk_size, :, :])
        g = torch.tanh(gates[:, 3*chunk_size:4*chunk_size, :, :])
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

# 2) ConvLSTM Wrapper
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1, batch_first=True, return_sequences=False):
        super(ConvLSTM, self).__init__()
        if num_layers != 1: logger.warning("This ConvLSTM class instance represents a single layer. num_layers parameter ignored.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.cell = ConvLSTMCell(input_dim=self.input_dim, hidden_dim=self.hidden_dim, kernel_size=self.kernel_size)

    def forward(self, x, hidden=None):
        if not self.batch_first: x = x.permute(1, 0, 2, 3, 4)
        b, seq_len, _, h, w = x.size()
        if hidden is None: hidden = self.cell.init_hidden(b, (h, w))
        h_cur, c_cur = hidden
        outputs = []
        for t in range(seq_len):
            current_input = x[:, t, :, :, :]
            h_cur, c_cur = self.cell(current_input, (h_cur, c_cur))
            if self.return_sequences: outputs.append(h_cur.unsqueeze(1))
        if self.return_sequences:
            outputs = torch.cat(outputs, dim=1)
            return outputs, (h_cur, c_cur)
        else:
            return h_cur, (h_cur, c_cur)

# 3) FrameDataset
class FrameDataset(Dataset):
    def __init__(self, data_path, img_size=(64, 64), time_steps_in=10, time_steps_out=50):
        super().__init__()
        self.data_path = data_path
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        self.total_time_steps = time_steps_in + time_steps_out
        self.img_size = img_size

        logger.info(f"Searching for images in: {self.data_path}")
        try:
            all_files = sorted([
                f for f in os.listdir(data_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
            ])
        except FileNotFoundError:
            logger.error(f"Data path not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error listing files in {self.data_path}: {e}")
            raise

        if not all_files:
            logger.error(f"No compatible image files found in {data_path}.")
            raise FileNotFoundError(f"No image files found in {data_path}")
        logger.info(f"Found {len(all_files)} potential image files.")

        self.transform_ops = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.images = []
        logger.info(f"Loading and transforming images (target size: {self.img_size})...")
        loaded_count = 0
        skipped_count = 0
        for i, fname in enumerate(all_files):
            img_path = os.path.join(data_path, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                img_t = self.transform_ops(img)
                self.images.append(img_t)
                loaded_count += 1
            except FileNotFoundError:
                logger.warning(f"Image file not found during loading: {img_path}")
                skipped_count += 1
            except Exception as e:
                logger.warning(f"Could not load or transform image '{fname}'. Error: {e}")
                skipped_count += 1
            finally:
                if (i+1) % 200 == 0:
                    logger.info(f"Processed {i+1}/{len(all_files)} files (Loaded: {loaded_count}, Skipped: {skipped_count})...")

        logger.info(f"Finished loading images. Total loaded: {loaded_count}, Total skipped: {skipped_count}.")

        if len(self.images) < self.total_time_steps:
            err_msg = f"Not enough valid images loaded ({len(self.images)}). Need at least {self.total_time_steps}."
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.images = torch.stack(self.images, dim=0)
        self.valid_length = len(self.images) - self.total_time_steps + 1
        logger.info(f"Total usable images: {len(self.images)}")
        logger.info(f"Number of valid sequences: {self.valid_length}")

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        if not 0 <= idx < self.valid_length:
            logger.error(f"Index {idx} out of bounds for dataset with length {self.valid_length}")
            raise IndexError(f"Index {idx} out of bounds")
        try:
            start_idx_x = idx
            end_idx_x = idx + self.time_steps_in
            X = self.images[start_idx_x : end_idx_x]
            start_idx_y = end_idx_x
            end_idx_y = start_idx_y + self.time_steps_out
            y = self.images[start_idx_y : end_idx_y]
            return X, y
        except Exception as e:
            logger.exception(f"Error getting item at index {idx}: {e}")
            raise

# 4) Seq2SeqConvLSTM Model
class Seq2SeqConvLSTM(nn.Module):
    def __init__(self, num_channels=3, hidden_dim=64,
                 encoder_layers=2, decoder_layers=2, kernel_size=3,
                 img_height=64, img_width=64, time_steps_out=50):
        super(Seq2SeqConvLSTM, self).__init__()
        self.time_steps_out_trained = time_steps_out
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.decoder_layers = decoder_layers

        self.encoder_stack = nn.ModuleList()
        current_dim = num_channels
        for i in range(encoder_layers):
            is_last_encoder_layer = (i == encoder_layers - 1)
            self.encoder_stack.append(
                ConvLSTM(input_dim=current_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                         num_layers=1, batch_first=True, return_sequences=not is_last_encoder_layer)
            )
            current_dim = hidden_dim

        self.decoder_stack = nn.ModuleList()
        current_dim = hidden_dim
        for _ in range(decoder_layers):
            self.decoder_stack.append(
                ConvLSTM(input_dim=current_dim, hidden_dim=hidden_dim, kernel_size=kernel_size,
                         num_layers=1, batch_first=True, return_sequences=True)
            )
            current_dim = hidden_dim

        self.output_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=1, padding=0)

    def forward(self, x_input, future_target=None, teacher_forcing_ratio=0.0):
        try:
            b, _, _, h, w = x_input.size()
            encoder_output = x_input
            encoder_final_states = []
            for layer in self.encoder_stack:
                encoder_output, last_hidden = layer(encoder_output)
                encoder_final_states.append(last_hidden)

            decoder_hidden_states = []
            num_encoder_states = len(encoder_final_states)
            for i in range(self.decoder_layers):
                encoder_state_idx = min(i, num_encoder_states - 1)
                decoder_hidden_states.append(encoder_final_states[encoder_state_idx])

            predictions = []
            decoder_input_step_h = encoder_output.unsqueeze(1)

            for t in range(self.time_steps_out_trained):
                current_decoder_input = decoder_input_step_h
                for i, layer in enumerate(self.decoder_stack):
                    current_decoder_output_h, next_hidden = layer(current_decoder_input, decoder_hidden_states[i])
                    decoder_hidden_states[i] = next_hidden
                    current_decoder_input = current_decoder_output_h

                output_frame = self.output_conv(current_decoder_output_h.squeeze(1))
                predictions.append(output_frame.unsqueeze(1))

                use_teacher_forcing = True if (self.training and future_target is not None and
                                              random.random() < teacher_forcing_ratio) else False

                if use_teacher_forcing:
                    decoder_input_step_h = current_decoder_output_h
                else:
                    decoder_input_step_h = current_decoder_output_h

            predictions = torch.cat(predictions, dim=1)
            predictions = torch.sigmoid(predictions)
            return predictions
        except Exception as e:
            logger.exception("Error during model forward pass:")
            raise

# --------------------------------------------------
# Metric Calculation Functions
# --------------------------------------------------
def calculate_mae(pred, target):
    """Calculate Mean Absolute Error (MAE) between predictions and target."""
    return torch.mean(torch.abs(pred - target)).item()

def calculate_psnr(pred, target, max_pixel_value=1.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between predictions and target."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')  # Perfect prediction
    return 10 * torch.log10((max_pixel_value ** 2) / mse).item()

# --------------------------------------------------
# Main Training Script
# --------------------------------------------------
def main():
    global logger, device

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_output_dir = BASE_OUTPUT_DIR.format(timestamp=run_timestamp)

    log_dir = os.path.join(base_output_dir, LOGS_DIR_NAME)
    checkpoint_dir = os.path.join(base_output_dir, CHECKPOINTS_DIR_NAME)
    plot_dir = os.path.join(base_output_dir, PLOTS_DIR_NAME)

    try:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directories: {e}")
        sys.exit(1)

    logger = setup_logging(log_dir, run_timestamp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"--- Training Configuration ---")
    logger.info(f"Using device: {device}")
    logger.info(f"Input Data Path: {DATA_PATH}")
    logger.info(f"Output Base: {base_output_dir}")
    logger.info(f"Input Steps: {TIME_STEPS_IN}, Output Steps: {TIME_STEPS_OUT}")
    logger.info(f"Image Size: ({IMG_HEIGHT}x{IMG_WIDTH}), Channels: {NUM_CHANNELS}")
    logger.info(f"Batch Size: {BATCH_SIZE}, Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    logger.info(f"Hidden Dim: {HIDDEN_DIM}, Encoder Layers: {ENCODER_LAYERS}, Decoder Layers: {DECODER_LAYERS}")
    logger.info(f"Teacher Forcing Ratio: {TEACHER_FORCING_RATIO}, Grad Clip: {GRADIENT_CLIPPING_VALUE}")
    logger.info(f"------------------------------\n")

    start_time = time.time()

    try:
        logger.info("Initializing Dataset and DataLoader...")
        dataset = FrameDataset(
            data_path=DATA_PATH,
            img_size=(IMG_HEIGHT, IMG_WIDTH),
            time_steps_in=TIME_STEPS_IN,
            time_steps_out=TIME_STEPS_OUT
        )
        train_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=True
        )
        total_batches = len(train_loader)
        logger.info(f"DataLoader created with {total_batches} batches.")
        if total_batches == 0:
            logger.error("DataLoader has zero batches. Check dataset size and batch size.")
            sys.exit(1)
    except FileNotFoundError:
        sys.exit(1)
    except ValueError as e:
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error during Dataset/DataLoader initialization:")
        sys.exit(1)

    try:
        logger.info("Initializing model...")
        model = Seq2SeqConvLSTM(
            num_channels=NUM_CHANNELS, hidden_dim=HIDDEN_DIM, encoder_layers=ENCODER_LAYERS,
            decoder_layers=DECODER_LAYERS, kernel_size=KERNEL_SIZE, img_height=IMG_HEIGHT,
            img_width=IMG_WIDTH, time_steps_out=TIME_STEPS_OUT
        ).to(device)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized on {device}. Trainable Parameters: {param_count:,}")
    except Exception as e:
        logger.exception("Fatal error during model initialization:")
        sys.exit(1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, verbose=False
    )

    train_losses = []
    train_maes = []
    train_psnrs = []
    logger.info("\n--- Starting Training ---")
    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            model.train()
            epoch_train_loss = 0.0
            epoch_train_mae = 0.0
            epoch_train_psnr = 0.0
            processed_batches = 0

            for i, batch_data in enumerate(train_loader):
                if batch_data is None:
                    logger.warning(f"Skipping invalid batch data at index {i}")
                    continue

                batch_start_time = time.time()
                try:
                    X, y = batch_data
                    X = X.to(device, non_blocking=PIN_MEMORY)
                    y = y.to(device, non_blocking=PIN_MEMORY)

                    optimizer.zero_grad()
                    pred = model(X, y, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
                    loss = criterion(pred, y)
                    mae = calculate_mae(pred, y)
                    psnr = calculate_psnr(pred, y)

                    loss.backward()
                    if GRADIENT_CLIPPING_VALUE > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIPPING_VALUE)
                    optimizer.step()

                    current_loss = loss.item()
                    current_mae = mae
                    current_psnr = psnr
                    epoch_train_loss += current_loss
                    epoch_train_mae += current_mae
                    epoch_train_psnr += current_psnr
                    processed_batches += 1
                    batch_end_time = time.time()

                    if (i + 1) % (max(1, total_batches // 10)) == 0:
                        logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{total_batches}], "
                                    f"Loss: {current_loss:.6f}, MAE: {current_mae:.6f}, PSNR: {current_psnr:.2f}dB, "
                                    f"Time: {batch_end_time - batch_start_time:.2f}s")

                except Exception as batch_e:
                    logger.exception(f"Error processing batch {i+1} in epoch {epoch+1}:")

            if processed_batches == 0:
                logger.warning(f"Epoch {epoch+1} completed without processing any batches. Check data.")
                continue

            avg_epoch_train_loss = epoch_train_loss / processed_batches
            avg_epoch_train_mae = epoch_train_mae / processed_batches
            avg_epoch_train_psnr = epoch_train_psnr / processed_batches
            train_losses.append(avg_epoch_train_loss)
            train_maes.append(avg_epoch_train_mae)
            train_psnrs.append(avg_epoch_train_psnr)
            epoch_end_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} Summary ---")
            logger.info(f"Average Train Loss: {avg_epoch_train_loss:.6f}, MAE: {avg_epoch_train_mae:.6f}, "
                        f"PSNR: {avg_epoch_train_psnr:.2f}dB")
            logger.info(f"Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")
            logger.info(f"Current Learning Rate: {current_lr:.6e}")

            scheduler.step(avg_epoch_train_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < current_lr:
                logger.info(f"Learning rate reduced by scheduler to {new_lr:.6e}")

            if (epoch + 1) % SAVE_CHECKPOINT_EPOCH_INTERVAL == 0 or epoch == NUM_EPOCHS - 1:
                checkpoint_name = f"model_epoch_{epoch+1}_loss_{avg_epoch_train_loss:.4f}.pth"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_epoch_train_loss,
                        'mae': avg_epoch_train_mae,
                        'psnr': avg_epoch_train_psnr,
                        'config': {
                            'IMG_HEIGHT': IMG_HEIGHT, 'IMG_WIDTH': IMG_WIDTH, 'NUM_CHANNELS': NUM_CHANNELS,
                            'TIME_STEPS_IN': TIME_STEPS_IN, 'TIME_STEPS_OUT': TIME_STEPS_OUT,
                            'HIDDEN_DIM': HIDDEN_DIM, 'ENCODER_LAYERS': ENCODER_LAYERS, 'DECODER_LAYERS': DECODER_LAYERS,
                            'KERNEL_SIZE': KERNEL_SIZE
                        }
                    }, checkpoint_path)
                    logger.info(f"Model checkpoint saved to {checkpoint_path}")
                except Exception as ckpt_e:
                    logger.exception(f"Error saving checkpoint at epoch {epoch+1}:")

            logger.info(f"-------------------------\n")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (KeyboardInterrupt).")
    except Exception as train_e:
        logger.exception("An unexpected error occurred during the training loop:")
    finally:
        total_training_time = time.time() - start_time
        logger.info(f"--- Training Finished ---")
        logger.info(f"Total Training Time: {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.2f}s")

    try:
        logger.info("Plotting training loss...")
        loss_plot_filename = os.path.join(plot_dir, f"training_loss_{run_timestamp}.png")
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='.', linestyle='-')
        plt.xlabel("Epochs")
        plt.ylabel("Average MSE Loss")
        plt.title(f"Training Loss Curve (Predicting {TIME_STEPS_OUT} steps)")
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(loss_plot_filename)
        plt.close()
        logger.info(f"Training loss plot saved to {loss_plot_filename}")
    except Exception as plot_e:
        logger.exception("Error generating or saving loss plot:")

    if 'X' in locals() and 'y' in locals() and 'pred' in locals():
        try:
            logger.info("Generating prediction visualization from last trained batch...")
            model.eval()
            X_vis = X[0:1]
            y_vis = y[0:1]
            pred_vis = pred[0:1]
            X_vis_np = X_vis[0].cpu().detach().numpy()
            y_vis_np = y_vis[0].cpu().detach().numpy()
            pred_vis_np = pred_vis[0].cpu().detach().numpy()
            X_vis_plot = np.transpose(X_vis_np, (0, 2, 3, 1))
            y_vis_plot = np.transpose(y_vis_np, (0, 2, 3, 1))
            pred_vis_plot = np.transpose(pred_vis_np, (0, 2, 3, 1))
            pred_vis_plot = np.clip(pred_vis_plot, 0, 1)
            num_in_frames = X_vis_plot.shape[0]
            num_out_frames = y_vis_plot.shape[0]
            input_indices = np.linspace(0, num_in_frames - 1, 3, dtype=int)
            output_indices = np.linspace(0, num_out_frames - 1, 5, dtype=int)
            num_plot_rows = 3
            num_plot_cols = max(len(input_indices), len(output_indices))
            plt.figure(figsize=(num_plot_cols * 3, num_plot_rows * 3))
            for i, frame_idx in enumerate(input_indices):
                plt.subplot(num_plot_rows, num_plot_cols, i + 1)
                plt.imshow(X_vis_plot[frame_idx])
                plt.title(f"Input t-{num_in_frames - 1 - frame_idx}")
                plt.axis("off")
            for i, frame_idx in enumerate(output_indices):
                plt.subplot(num_plot_rows, num_plot_cols, num_plot_cols + i + 1)
                plt.imshow(y_vis_plot[frame_idx])
                plt.title(f"True t+{frame_idx + 1}")
                plt.axis("off")
            for i, frame_idx in enumerate(output_indices):
                plt.subplot(num_plot_rows, num_plot_cols, 2 * num_plot_cols + i + 1)
                plt.imshow(pred_vis_plot[frame_idx])
                plt.title(f"Pred t+{frame_idx + 1}")
                plt.axis("off")
            plt.suptitle(f"Sample Prediction Vis. (Epoch {NUM_EPOCHS})")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            vis_plot_filename = os.path.join(plot_dir, f"prediction_vis_{run_timestamp}.png")
            plt.savefig(vis_plot_filename)
            plt.close()
            logger.info(f"Prediction visualization saved to {vis_plot_filename}")
        except Exception as vis_e:
            logger.exception("Error generating or saving prediction visualization:")
    else:
        logger.warning("Skipping visualization as last batch data is not available.")

    logger.info("Script finished.")

if __name__ == "__main__":
    main()