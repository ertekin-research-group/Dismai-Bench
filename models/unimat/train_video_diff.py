import os
import argparse
import time
import torch
from imagen_pytorch import Unet3D, Imagen, ImagenTrainer
from math import ceil
import glob
import csv
import logging


def imagen_model(args):
    """
    Constructs an Imagen model
    """
    unet = Unet3D(
                  dim = args.dim, 
                  dim_mults = (1, 2, 4),
                  num_resnet_blocks = 3,
                  layer_attns = (False, True, True),
                  layer_cross_attns = (False, True, True)
                  )

    imagen = Imagen(
                    condition_on_text = False,
                    unets = unet,
                    image_sizes = args.img_size,
                    timesteps = 1000,
                    channels = 3,
                    auto_normalize_img = False
                    )
    
    return imagen

class DataloaderMeter(object):
    "Tracks the current batch size of the Dataloader"

    def __init__(self, batch_size, n_total):
        self.batch_size = batch_size
        self.n_total = n_total
        self.reset()

    def reset(self):
        self.n_remaining = self.n_total

    def update(self):
        self.n_remaining -= self.batch_size
        if self.n_remaining >= 0:
            self.current_batch_size = self.batch_size
        else:
            self.current_batch_size = self.batch_size + self.n_remaining
        if self.n_remaining <= 0:
            self.reset()

class AverageMeter(object):
    "Computes and stores the average and current value"

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/path/to/data/train_val_data/unimat/int', help='path to UniMat dataset directory')
    parser.add_argument('--n_steps', type=int, default=150001, help='number of training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--chunk_size', type=int, default=8, help='maximum size of chunks each mini-batch is split into')
    parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam: beta_1')
    parser.add_argument('--beta2', type=float, default=0.99, help='adam: beta_2')
    parser.add_argument('--dim', type=int, default=64, help='number of base channels of the U-Net')
    parser.add_argument('--save_interval', type=int, default=150, help='checkpoint and losses are saved every `save_interval` training steps')
    parser.add_argument('--val_once_every_k', type=int, default=5, help='validation is run once every `val_once_every_k`*`save_interval`')
    parser.add_argument('--save_dir', type=str, default='./model_saves', help="directory to save checkpoint")
    parser.add_argument('--load_ckpt', type=str, help='path to checkpoint to be loaded')
    parser.add_argument('--print_interval', type=int, default=150, help='output is printed every `print_interval` steps')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(filename='log.txt', filemode='a', format='%(message)s',
                        level=logging.INFO)
    logging.info(args)
    
    # Set device
    """
    Due to the way ImagenTrainer is coded, it will always use gpu if available.
    To use cpu, you need to hide the gpu from PyTorch before running the training script.
    If running the script in command line:
        export CUDA_VISIBLE_DEVICES=','
    If running the script in an IDE:
        import os; os.environ['CUDA_VISIBLE_DEVICES'] = ','
    """
    if torch.cuda.is_available():
        logging.info('Running on gpu')
    else:
        logging.info('Running on cpu')
    
    ## Initialize minimum validation loss and starting step number
    min_val_loss = 1e10
    start_step = 0
    
    # Create directory for saving trained models
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load data
    logging.info('Loading data...')
    data_train = torch.load(args.data_dir + '/unimat_train.pt')
    logging.info(f'=> Loaded `{args.data_dir}/unimat_train.pt`')
    data_val = torch.load(args.data_dir + '/unimat_val.pt')
    logging.info(f'=> Loaded `{args.data_dir}/unimat_val.pt`')
    args.img_size = data_train.shape[-1]
    
    # Initialize model and trainer
    logging.info('Initializing model and trainer...')
    imagen = imagen_model(args)
    
    trainer = ImagenTrainer(imagen, 
                            lr=args.lr, 
                            beta1=args.beta1, 
                            beta2=args.beta2
                            )
    trainer.add_train_dataset(data_train, batch_size=args.batch_size, shuffle=True)
    trainer.add_valid_dataset(data_val, batch_size=args.batch_size, shuffle=True)
    
    # Load model if provided
    if args.load_ckpt:
        trainer.load(args.load_ckpt)
        # Resume training - get last step number and min_val_loss from previous training
        with open('losses.csv', 'r') as file:
            last_line = file.readlines()[-1]
        start_step = int(last_line.split(',')[0]) + 1
        min_val_loss = float(last_line.split(',')[-1].strip())
        logging.info(f'=> Loaded `{args.load_ckpt}`')
    else:
        logging.info('=> Model is initialized from scratch')
    
    logging.info(f'Starting training from step {start_step}!')
    n_val = data_val.shape[0]   # Total number of validation samples
    n_val_batch = ceil(n_val/args.batch_size)   # Number of validation steps (batches) to go through entire validation set once
    meter_dl_val = DataloaderMeter(args.batch_size, n_val)
    n_train = data_train.shape[0]   # Total number of training samples
    if n_train % args.batch_size != 0:
        logging.info('Warning: The number of training samples is not divisible by the batch size, '
                     'the moving average training loss will have some error')
    
    meter_loss_train = AverageMeter()   # Stores the moving average of the training loss for every `save_interval`
    meter_batch_time = AverageMeter()   # Stores the batch time
    for step_num in range(start_step, args.n_steps):
        # Perform one training step
        end = time.time()   # Time stamp
        loss = trainer.train_step(unet_number=1, 
                                  max_batch_size=args.chunk_size)
        meter_loss_train.update(loss)
        
        meter_batch_time.update(time.time() - end)
        if step_num % args.print_interval == 0:
            logging.info(f'Step: [{step_num}]\t'
                         f'Batch time {meter_batch_time.value:.3f} ({meter_batch_time.avg:.3f})\t'
                         f'Train loss {meter_loss_train.value:.6f} ({meter_loss_train.avg:.6f})\t'
                         f'Memory used {torch.cuda.max_memory_allocated()/2**30:.3f}\t'
                         f'Memory reserved {torch.cuda.max_memory_reserved()/2**30:.3f}')
        
        if step_num % (args.val_once_every_k * args.save_interval) == 0 and step_num != 0:
            meter_loss_val = AverageMeter()   # Stores the validation loss
            # Perform validation steps over the entire validation dataset
            for i_val in range(n_val_batch):
                meter_dl_val.update()
                current_batch_size = meter_dl_val.current_batch_size
                loss = trainer.valid_step(unet_number=1, 
                                          max_batch_size=args.chunk_size)
                meter_loss_val.update(loss, n=current_batch_size)
            
            if meter_loss_val.avg < min_val_loss:
                min_val_loss = meter_loss_val.avg
                # Save best checkpoint
                previous_best_ckpt = glob.glob(args.save_dir+'/*best.pt')
                save_fname = args.save_dir+'/checkpoint_step='+str(step_num)+'_best.pt'
                trainer.save(save_fname)
                logging.info(f'=> Saved `{save_fname}`')
                if len(previous_best_ckpt) > 0:
                    os.remove(previous_best_ckpt[0])
                
        if step_num % args.save_interval == 0:
            # Save last checkpoint
            previous_ckpt = glob.glob(args.save_dir+'/*last.pt')
            save_fname = args.save_dir+'/checkpoint_step='+str(step_num)+'_last.pt'
            trainer.save(save_fname)
            logging.info(f'=> Saved `{save_fname}`')
            if len(previous_ckpt) > 0:
                os.remove(previous_ckpt[0])
        
            # Write losses
            with open("losses.csv", "a", newline='') as csvfile: 
                csvwriter = csv.writer(csvfile)
                if step_num == 0:
                    csvwriter.writerow(['step', 'train_loss_step', 'train_loss_moving_avg', 'val_loss', 'min_val_loss'])
                if step_num % (args.val_once_every_k * args.save_interval) != 0 or step_num == 0:
                    val_loss = None
                else:
                    val_loss = meter_loss_val.avg
                csvwriter.writerow([step_num, meter_loss_train.value, meter_loss_train.avg, val_loss, min_val_loss])
            
            meter_loss_train.reset()
    

if __name__ == '__main__':
    main()




