import torch
import logging
from contextlib import nullcontext
from pathlib import Path
import hydra
import wandb
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from model import VAE, VAEConfig
from data import init_dataloaders
from utils import (
    torch_compile_ckpt_fix,
    torch_get_device,
    torch_set_seed,
    get_ist_time_now,
    timer,
    AverageMetrics,
    WandBLogger,
    make_grid_and_save
)
OmegaConf.register_new_resolver("now_ist", get_ist_time_now)

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg):
    logger = logging.getLogger("vae")
    device = torch_get_device(cfg.device_type)
    logger.info(f"Using {device}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = Path(hydra_cfg.runtime.output_dir)
    run_name = hydra_cfg.job.name
    torch_autocast_dtype = {'f32': torch.float32, 'bf16': torch.bfloat16}[cfg.autocast_dtype]

    torch_set_seed(cfg.rng_seed)

    train_dataloader = init_dataloaders(cfg, split='train')
    val_dataloader = init_dataloaders(cfg, split = 'val')

    start_epoch = 1
    wandb_id = None
    if cfg.init_from == 'scratch':
        model_cfg = VAEConfig(**cfg.model)
        model = VAE(model_cfg)
        model.to(device)
    else:
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        ckpt_cfg = ckpt['config']
        model_cfg = VAEConfig(**ckpt_cfg.model)
        model = VAE(model_cfg)
        model.to(device)
        model.load_state_dict(torch_compile_ckpt_fix(ckpt['model']))
        logger.info(f"Loaded checkpoint from {cfg.init_from}")
        start_epoch = ckpt['epoch'] + 1
        wandb_id = ckpt.get('wandb_id')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_cfg.name} | Params: {total_params / 1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
    if cfg.torch_compile:
        model = torch.compile(model, dynamic=True)

    # optimizer
    optimizer = model.configure_optimizer(cfg.optimizer, device=device)
    if cfg.init_from != "scratch":
        optimizer.load_state_dict(ckpt['optimizer'])

    if cfg.enable_tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_autocast_dtype)
        if device.type == "cuda" and torch_autocast_dtype == torch.bfloat16
        else nullcontext()
    )

    wb_logger = WandBLogger(
        project=cfg.logging.wandb.project,
        run=run_name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        tags=("train", "val"),
        metrics=("loss", "recon_loss", "kl_loss", "time"),
        run_id=wandb_id,
        enable=cfg.logging.wandb.enable
    )

    @timer
    def train_epoch():
        model.train()
        metrics = AverageMetrics()
        progress_bar = tqdm(train_dataloader, dynamic_ncols=True, desc="Train", leave=False, disable=(not cfg.interactive))
        for step, batch in enumerate(progress_bar):
            imgs, lbls = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                imgs_pred, losses = model(imgs)
            losses['loss'].backward()
            if cfg.clip_grad_norm_1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optimizer.step()

            batch_size = imgs.size(0)
            metrics.update({ k:v.item() for k, v in losses.items() }, batch_size)
            progress_bar.set_postfix({ k:f"{v.item():.4f}" for k, v in losses.items() })
        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return metrics.compute()

    @timer
    @torch.no_grad()
    def val_epoch():
        model.eval()
        metrics = AverageMetrics()
        progress_bar = tqdm(val_dataloader, dynamic_ncols=True, desc="Val", leave=False, disable=(not cfg.interactive))
        for step, batch in enumerate(progress_bar):
            imgs, lbls = batch[0].to(device), batch[1].to(device)

            with autocast_ctx:
                imgs_pred, losses = model(imgs)

            batch_size = imgs.size(0)
            metrics.update({ k:v.item() for k, v in losses.items() }, batch_size)
            progress_bar.set_postfix({ k:f"{v.item():.4f}" for k, v in losses.items() })
        progress_bar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return metrics.compute()

    @torch.no_grad()
    def generate_reconstructions(epoch: int):
        model.eval()

        batch = next(iter(val_dataloader))
        imgs = batch[0].to(device)

        with autocast_ctx:
            imgs_pred, _ = model(imgs)

        max_imgs = min(16, imgs.size(0))
        imgs = imgs[:max_imgs]
        imgs_pred = imgs_pred[:max_imgs]

        combined = torch.cat([imgs, imgs_pred], dim=3)
        save_path = log_dir / f"reconstruction_{epoch}.png"
        grid = make_grid_and_save(combined, save_path)

        if wb_logger.enable:
            wandb.log({
                'epoch': epoch,
                'reconstruction': wandb.Image(grid)
            })

    t, stats = val_epoch()
    logger.info(f"Initial Val " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + f" Time={t:.2f}s")
    wb_logger.log("val", {'epoch': start_epoch-1, **stats, 'time':t})

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        last_epoch = epoch == cfg.n_epochs
        logger.info(f"Epoch: {epoch}/{cfg.n_epochs}")

        t, stats = train_epoch()
        logger.info(f"{'Train':<5} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + f" Time={t:.2f}s")
        wb_logger.log("train", {'epoch': epoch, **stats, 'time':t})

        if last_epoch or epoch % cfg.val_epoch_interval == 0:
            t, stats = val_epoch()
            logger.info(f"{'Val':<5} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()) + f" Time={t:.2f}s")
            wb_logger.log("val", {'epoch': epoch, **stats, 'time':t})

        if last_epoch or epoch % cfg.viz_epoch_interval == 0:
            generate_reconstructions(epoch)

        if last_epoch or epoch % cfg.save_epoch_interval == 0:
            ckpt_path = log_dir / f"{model_cfg.name}.pt"
            torch.save({
                'model': model.state_dict(),
                'config': cfg,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'wandb_id': wb_logger.run_id,
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {str(ckpt_path)}")

if __name__ == "__main__":
    main()