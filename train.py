import os
import argparse
import sys
from torchmetrics import MeanMetric
import torch
from data_utils import UnderwaterDataset, UnderwaterDatasetNonRef
from torch.utils.data import DataLoader
from model import myModel
from uranker import build_model, get_option
from train_utils import WandbLogger, visualize_results, visualize_results_non_ref, train_step, eval_step, CheckpointSaver, compute_metrics, reset_metrics
from torch import optim
from loss_utils import PerceptualLoss, L1_Charbonnier_loss, SSIMLoss, EdgeAwareLoss, CIDNet

def parse_args():
    parser = argparse.ArgumentParser(description="Train the underwater image enhancement model.")
    parser.add_argument('--train_degrade_dir', type=str, required=True, help='Directory containing degraded images.')
    parser.add_argument('--train_ref_dir', type=str, required=True, help='Directory containing reference images.')
    parser.add_argument('--val_degrade_dir', type=str, required=True, help='Directory containing degraded validation images.')
    parser.add_argument('--val_ref_dir', type=str, required=True, help='Directory containing reference validation images.')
    parser.add_argument('--non_ref_dir', type=str, required=True, help='Directory containing non-reference images.')
    parser.add_argument('--img_size', type=int, default=256, help='Size to which images will be resized/cropped.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--start_epochs', type=int, default=0, help='Starting epoch number (useful for resuming training).')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='Number of steps to accumulate gradients before updating model weights.')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Number of epochs between evaluations.')
    parser.add_argument('--sample_interval', type=int, default=1000, help='Number of batches between result visualizations.')
    parser.add_argument('--vis_interval', type=int, default=1000, help='Number of batches between visualizations.')
    parser.add_argument('--eval_interval', type=int, default=5, help='Number of epochs between evaluations.')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Number of epochs between saving model checkpoints. Set to -1 to disable.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu").')
    parser.add_argument('--wandb_project', type=str, default='WWE-UIE', help='WandB project name for logging.')
    parser.add_argument('--wandb_api_key', type=str, required=True, help='WandB API key for logging.')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--uranker_config_path', type=str, required=True, help='Path to the URanker configuration file for non-reference evaluation.')
    parser.add_argument('--uranker_ckpt_path', type=str, required=True, help='Path to the URanker checkpoint for non-reference evaluation.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    DEVICE = args.device
    training_metrics = {
        'L1 Loss': MeanMetric().to(DEVICE),
        'PSNR': MeanMetric().to(DEVICE),
        'SSIM': MeanMetric().to(DEVICE),
        'HVI Loss': MeanMetric().to(DEVICE),
        'VGG Loss': MeanMetric().to(DEVICE),
        'Edge Loss': MeanMetric().to(DEVICE),
        'Total Loss': MeanMetric().to(DEVICE)
    }
    eval_metrics = training_metrics.copy()

    train_dataset = UnderwaterDataset(args.train_degrade_dir, args.train_ref_dir, args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = UnderwaterDataset(args.val_degrade_dir, args.val_ref_dir, args.img_size, train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    non_ref_dataset = UnderwaterDatasetNonRef(args.non_ref_dir, args.img_size)
    non_ref_loader = DataLoader(non_ref_dataset, batch_size=args.batch_size, shuffle=False)

    model = myModel(in_channels=3, feature_channels=32, use_white_balance=True).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.n_epochs, eta_min=args.learning_rate * 1e-4
    )

    options = get_option(args.uranker_config_path)
    options["model"]["resume_ckpt_path"] = args.uranker_ckpt_path
    uranker_model = build_model(options["model"])

    wandb_logger = WandbLogger(project_name=args.wandb_project, api_key=args.wandb_api_key)

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_manager = CheckpointSaver(save_dir=args.save_dir, max_ckpt=5)

    model_dict = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}

    losses = {}
    losses['vggL'] = PerceptualLoss()
    losses['L1L'] = L1_Charbonnier_loss()
    losses['ssimL'] = SSIMLoss(device="cuda", window_size=5)
    losses['edgeL'] = EdgeAwareLoss(loss_type="l2", device="cuda")

    losses['hvi_net'] = CIDNet().cuda()
    pth = r"./CIDNet_weight_LOLv2_bestSSIM.pth"
    losses['hvi_net'].load_state_dict(torch.load(pth, map_location="cuda"))
    losses['hvi_net'].eval()

    for epoch in range(args.start_epochs, args.n_epochs):
    
        psnr_list = []
        reset_metrics(training_metrics)
        
        for i, batch in enumerate(train_loader):

            psnr_train, ssim_value, final_loss, hvi_loss, vgg_loss, edge_loss, batches_done = train_step(model, 
                                                                                                         train_loader, 
                                                                                                         batch, 
                                                                                                         i, 
                                                                                                         training_metrics,
                                                                                                         losses,
                                                                                                         optimizer,
                                                                                                         epoch,
                                                                                                         args.n_epochs,
                                                                                                         args.grad_accum_steps,
                                                                                                         DEVICE)

            # Print log
            if batches_done % 100==0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d][PSNR: %2f] [SSIM: %2f][loss: %2f][loss_lch: %2f][loss_lab: %2f][fdl_loss: %2f]"
                    % (
                        epoch,
                        args.n_epochs,
                        i,
                        len(train_loader),
                        psnr_train,
                        ssim_value,
                        final_loss.item(),
                        hvi_loss.item(),
                        vgg_loss.item(),
                        edge_loss.item(), 
                    )
                )

            if batches_done % args.sample_interval == 0:
                visualize_results(model, next(iter(val_loader)), 'Validation Results')
                visualize_results_non_ref(model, next(iter(non_ref_loader)), uranker_model, 'Non-ref Validation Results')
            psnr_list.append(psnr_train)
            
            train_results = compute_metrics(training_metrics)
            # print("\nTrain Metrics: " + " - ".join([f"{k}: {v:.4f}" for k, v in train_results.items()]))
            wandb_logger.log({f"Train/{k}": v for k, v in train_results.items()})

        if (epoch+1) % args.eval_interval == 0:
            reset_metrics(eval_metrics)
            for batch in val_loader:
                eval_step(model, batch, eval_metrics, losses, DEVICE)
            eval_results = compute_metrics(eval_metrics)
            # print("Eval Metrics: " + " - ".join([f"{k}: {v:.4f}" for k, v in eval_results.items()]))
            wandb_logger.log({f"Valid/{k}": v for k, v in eval_results.items()})

        scheduler.step()
        
        if args.checkpoint_interval != -1 and (epoch+1) % args.checkpoint_interval == 0:
            checkpoint_manager.save(
                model_dict=model_dict,
                epoch=epoch
            )