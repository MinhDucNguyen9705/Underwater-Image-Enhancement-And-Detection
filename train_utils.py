import os
import glob
import torch
import wandb
import datetime
from torchmetrics import MeanMetric
import matplotlib.pyplot as plt
from metrics import batch_PSNR, get_uciqe, calculate_niqe, getURanker
from loss_utils import SSIMLoss
from torch.autograd import Variable

class CheckpointSaver:
    """Saves model checkpoints during training based on specified criteria."""
    def __init__(self, save_dir='checkpoints', max_ckpt=5):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_ckpt = max_ckpt

    def save(self, model_dict, epoch):
        """Saves the model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, f'WWE_UIE_{epoch+1}.pth')
        # Create a clean dictionary for saving
        save_dict = {}
        for k, v in model_dict.items():
            if hasattr(v, "state_dict"):
                # if k == 'model':
                #     save_dict[k] = v.module.state_dict()
                # else:
                save_dict[k] = v.state_dict()
            else:
                save_dict[k] = v  # e.g., scalars, metrics, etc.
        save_dict['epoch'] = epoch
        torch.save(save_dict, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
        # Optionally limit number of saved checkpoints
        self._cleanup_old_checkpoints()

    def load(self, ckpt_path, model_dict=None):
        """Loads a model checkpoint and optionally restores model/optimizer states."""
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"Checkpoint loaded: {ckpt_path}")
        if model_dict is not None:
            for k, v in model_dict.items():
                if k in checkpoint and hasattr(v, "load_state_dict"):
                    v.load_state_dict(checkpoint[k])
                    print(f'Successfully loaded {k}')
        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove older checkpoints to maintain max_ckpt limit."""
        ckpts = sorted(glob.glob(os.path.join(self.save_dir, 'checkpoint_epoch_*.pt')), key=os.path.getmtime)
        if len(ckpts) > self.max_ckpt:
            for f in ckpts[:-self.max_ckpt]:
                os.remove(f)
                print(f"Old checkpoint removed: {f}")

class WandbLogger:
    """ Wandb Logger for logging metrics and images to Weights & Biases. """
    def __init__(self, project_name, api_key):
        self.project_name = project_name
        self.api_key = api_key
        self.init_run()

    def init_run(self):
        """ Initializes a new wandb run with the specified project name and API key. """
        wandb.login(key=self.api_key)
        self.run = wandb.init(project=self.project_name, name=str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    def log(self, dict):
        """ Logs a dictionary of metrics to wandb. """
        self.run.log(dict)

    def log_image(self, plot, caption: str):
        """ Logs an image to wandb with a caption. """
        self.run.log({caption: wandb.Image(plot, caption)})

def reset_metrics(metrics):
    """ Resets all metrics in the provided dictionary. """
    for metric in metrics.values():
        metric.reset()

def update_metrics(metrics, values_dict):
    """ Updates all metrics in the provided dictionary with new predictions and targets. """
    for k, v in values_dict.items():
        if k in metrics:
            metrics[k](v)

def compute_metrics(metrics):
    """ Computes and returns the current value of all metrics in the provided dictionary. """
    return {k: metric.compute().item() for k, metric in metrics.items()}

def visualize_results(model, batch, caption='Validation Results', device='cpu', wandb_logger=None):
    model.eval()
    
    degrade_img, ref_img = batch
    degrade_img = degrade_img.to(device)
    ref_img = ref_img.to(device)

    with torch.no_grad():
        pred_img = model(degrade_img)
        pred_img = torch.clamp(pred_img, 0., 1.)

    num_samples = min(4, degrade_img.shape[0])
    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axs = [axs]

    ssimL = SSIMLoss(device=device, window_size=5)

    for i in range(num_samples):
        axs[i][0].imshow(degrade_img[i].permute(1, 2, 0).cpu().numpy())
        axs[i][0].set_title('Degraded Image')
        axs[i][0].axis('off')

        psnr_value = batch_PSNR(pred_img[i].unsqueeze(0), ref_img[i].unsqueeze(0), 1.)
        ssim_value = 1 - ssimL(pred_img[i].unsqueeze(0), ref_img[i].unsqueeze(0))

        # in case PSNR/SSIM return tensors
        if torch.is_tensor(psnr_value):
            psnr_value = psnr_value.item()
        if torch.is_tensor(ssim_value):
            ssim_value = ssim_value.item()

        axs[i][1].imshow(pred_img[i].permute(1, 2, 0).cpu().numpy())
        axs[i][1].set_title(f'PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.2f}')
        axs[i][1].axis('off')

        axs[i][2].imshow(ref_img[i].permute(1, 2, 0).cpu().numpy())
        axs[i][2].set_title('Reference Image')
        axs[i][2].axis('off')

    plt.tight_layout()
    wandb_logger.log_image(fig, caption=caption)
    plt.show()

def visualize_results_non_ref(model, batch, uranker_model, caption='Validation Results', device='cpu', wandb_logger=None):
    model.eval()
    
    degrade_img = batch
    degrade_img = degrade_img.to(device)

    with torch.no_grad():
        pred_img = model(degrade_img)
        pred_img = torch.clamp(pred_img, 0., 1.)

    num_samples = min(4, degrade_img.shape[0])
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))

    if num_samples == 1:
        axs = [axs]

    for i in range(num_samples):
        image = degrade_img[i].permute(1, 2, 0).cpu().numpy()
        niqe_val = calculate_niqe(image[:, :, ::-1] * 255)
        uciqe_val = get_uciqe(image)
        uranker_val = getURanker(image[None, :], uranker_model)
        axs[i][0].imshow(degrade_img[i].permute(1, 2, 0).cpu().numpy())
        axs[i][0].set_title(f'NIQE: {niqe_val: .2f}, UCIQE: {uciqe_val: .2f}, URanker: {uranker_val: .2f}')
        axs[i][0].axis('off')

        image = pred_img[i].permute(1, 2, 0).cpu().numpy()
        niqe_val = calculate_niqe(image[:, :, ::-1] * 255)
        uciqe_val = get_uciqe(image)
        uranker_val = getURanker(image[None, :], uranker_model)
        axs[i][1].imshow(pred_img[i].permute(1, 2, 0).cpu().numpy())
        axs[i][1].set_title(f'NIQE: {niqe_val: .2f}, UCIQE: {uciqe_val: .2f}, URanker: {uranker_val: .2f}')
        axs[i][1].axis('off')

    plt.tight_layout()
    wandb_logger.log_image(fig, caption=caption)
    plt.show()

def train_step(model, train_loader, batch, step, training_metrics, losses, optimizer, epoch, n_epochs, grad_accum_steps, device):
    model.train()

    x = Variable(batch[0]).to(device).contiguous() 
    label = Variable(batch[1]).to(device).contiguous()

    pred = model(x)
    with torch.no_grad():
        label_hvi = losses['hvi_net'].trans.HVIT(label)
    pred_hvi = losses['hvi_net'].trans.HVIT(pred.clamp(0.0, 1.0))
    hvi_loss = losses['L1L'](pred_hvi, label_hvi)
    l1_loss = losses['L1L'](pred, label)
    vgg_loss = losses['vggL'](pred, label)
    ssim_loss = losses['ssimL'](pred, label)
    edge_loss = losses['edgeL'](pred, label)
    final_loss = (
        l1_loss
        + 0.5 * hvi_loss
        + 0.1 * ssim_loss
        + 0.1 * vgg_loss
        + 0.1 * edge_loss
    )

    final_loss = final_loss / grad_accum_steps
    final_loss.backward(retain_graph=True)

    if (step + 1) % grad_accum_steps == 0:
        # Optimized
        optimizer.step()
        optimizer.zero_grad()

    batches_done = epoch * len(train_loader) + step
    batches_left = n_epochs * len(train_loader) - batches_done
    out_train = torch.clamp(pred, 0., 1.) 
    psnr_train = batch_PSNR(out_train,label, 1.)

    met = {
        'L1 Loss': l1_loss.item(),
        'PSNR': psnr_train.item(),
        'SSIM': 1 - ssim_loss.item(),
        'HVI Loss': hvi_loss.item(),
        'VGG Loss': vgg_loss.item(),
        'Edge Loss': edge_loss.item(),
        'Total Loss': final_loss.item()
    }
    update_metrics(training_metrics, met)

    return psnr_train, 1 - ssim_loss, final_loss, hvi_loss, vgg_loss, edge_loss, batches_done

def eval_step(model, batch, eval_metrics, losses, device):
    model.eval()

    x = batch[0].to(device).contiguous() 
    label = batch[1].to(device).contiguous()

    with torch.no_grad():
        pred = model(x)
        label_hvi = losses['hvi_net'].trans.HVIT(label)
        pred_hvi = losses['hvi_net'].trans.HVIT(pred.clamp(0.0, 1.0))
        hvi_loss = losses['L1L'](pred_hvi, label_hvi)
        l1_loss = losses['L1L'](pred, label)
        vgg_loss = losses['vggL'](pred, label)
        ssim_loss = losses['ssimL'](pred, label)
        edge_loss = losses['edgeL'](pred, label)
        final_loss = (
            l1_loss
            + 0.5 * hvi_loss
            + 0.1 * ssim_loss
            + 0.1 * vgg_loss
            + 0.1 * edge_loss
        )
    
        out_train = torch.clamp(pred, 0., 1.) 
        psnr_train = batch_PSNR(out_train,label, 1.)
        met = {
            'L1 Loss': l1_loss.item(),
            'PSNR': psnr_train.item(),
            'SSIM': 1 - ssim_loss.item(),
            'HVI Loss': hvi_loss.item(),
            'VGG Loss': vgg_loss.item(),
            'Edge Loss': edge_loss.item(),
            'Total Loss': final_loss.item()
        }
        update_metrics(eval_metrics, met)