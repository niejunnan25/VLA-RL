from pathlib import Path

from omegaconf import OmegaConf
import torch

from examples.libero_rlt.train_stage1 import _build_modules, checkpoint_payload
from vla_rl.algorithms.rlt.features import load_frozen_rlt_encoder


def test_stage1_fake_prefix_reconstruction_step():
    cfg = OmegaConf.create(
        {
            "rlt": {
                "rl_token_dim": 8,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "num_heads": 2,
                "ff_dim": 16,
                "dropout": 0.0,
                "max_tokens": 3,
            },
            "training": {
                "steps": 2,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_steps": 0,
                "clip_grad_norm": 1.0,
            },
        }
    )
    encoder, decoder, optimizer, scheduler = _build_modules(cfg, input_dim=8, device=torch.device("cpu"))
    z_vla = torch.zeros(2, 3, 8)
    z_rl = encoder(z_vla)
    z_recon = decoder(z_rl, z_vla)
    loss = torch.nn.functional.mse_loss(z_recon, z_vla)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    assert z_rl.shape == (2, 8)
    assert z_recon.shape == z_vla.shape


def test_stage1_checkpoint_loads_in_stage2(tmp_path: Path):
    cfg = OmegaConf.create(
        {
            "rlt": {
                "rl_token_dim": 8,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "num_heads": 2,
                "ff_dim": 16,
                "dropout": 0.0,
                "max_tokens": 3,
            },
            "training": {
                "steps": 2,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_steps": 0,
            },
        }
    )
    encoder, decoder, optimizer, scheduler = _build_modules(cfg, input_dim=8, device=torch.device("cpu"))
    path = tmp_path / "final_model.pt"
    torch.save(checkpoint_payload(cfg, 8, 2, encoder, decoder, optimizer, scheduler), path)

    loaded = load_frozen_rlt_encoder(str(path), device="cpu")

    assert loaded(torch.zeros(1, 3, 8)).shape == (1, 8)
    assert getattr(loaded, "max_tokens") == 3
