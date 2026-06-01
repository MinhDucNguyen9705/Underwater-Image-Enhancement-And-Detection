import os
import argparse
from utils.data_utils import UnderwaterDataset, UnderwaterDatasetNonRef
from metrics import getURanker, batch_PSNR, calculate_niqe, get_uciqe
from utils.loss_utils import SSIMLoss
from torchmetrics import MeanMetric
from utils.uranker import build_model, get_option
from utils.train_utils import compute_metrics, reset_metrics


def args_parse():
    parser = argparse.ArgumentParser(description='Evaluate the model on the test set')
    parser.add_argument('--clean_dir', type=str, required=True, help='Directory of clean images')
    parser.add_argument('--ref_dir', type=str, default=None, help='Directory of reference images')
    parser.add_argument('--img_size', type=int, default=None, help='Size of the input images')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation (e.g., "cuda" or "cpu")')
    parser.add_argument('--uranker_config_path', type=str, required=True, help='Path to the URanker config file')
    parser.add_argument('--uranker_ckpt_path', type=str, required=True, help='Path to the URanker checkpoint file')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = args_parse()
    DEVICE = args.device

    options = get_option(args.uranker_config_path)
    options["model"]["resume_ckpt_path"] = args.uranker_ckpt_path
    uranker_model = build_model(options["model"])

    ssimL = SSIMLoss(device="cuda", window_size=5)
    
    if args.ref_dir is None:
        dataset = UnderwaterDatasetNonRef(args.clean_dir, args.img_size, train=False)
    else:
        dataset = UnderwaterDataset(args.clean_dir, args.ref_dir, args.img_size, train=False)
    print(f"Number of test images: {len(dataset)}")
    
    eval_metrics = {
        'UCIQE': MeanMetric().to(DEVICE),
        'NIQE': MeanMetric().to(DEVICE),
        'URanker': MeanMetric().to(DEVICE)
    }
    
    if args.ref_dir is not None:
        eval_metrics['PSNR'] = MeanMetric().to(DEVICE)
        eval_metrics['SSIM'] = MeanMetric().to(DEVICE)
    
    for i in range(len(dataset)):
        if args.ref_dir is not None:
            clean_img, ref_img = dataset[i]
            psnr = batch_PSNR(clean_img.unsqueeze(0), ref_img.unsqueeze(0), data_range=1.0)
            ssim_value = 1 - ssimL(clean_img.unsqueeze(0).to(DEVICE), ref_img.unsqueeze(0).to(DEVICE))
            eval_metrics['PSNR'].update(psnr)
            eval_metrics['SSIM'].update(ssim_value)
        else:
            clean_img = dataset[i]

        image = clean_img.permute(1, 2, 0).cpu().numpy()
        niqe_val = calculate_niqe(image[:, :, ::-1] * 255)
        uciqe_val = get_uciqe(image)
        uranker_val = getURanker(image[None, :], uranker_model)
        
        eval_metrics['UCIQE'].update(uciqe_val)
        eval_metrics['NIQE'].update(niqe_val)
        eval_metrics['URanker'].update(uranker_val)

    print("Evaluation Results:")
    for metric_name, metric in eval_metrics.items():
        print(f"{metric_name}: {metric.compute().item():.4f}")