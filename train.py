import argparse, os
import torch
from torch.utils.data import DataLoader
from src.data.datasets import HSICDPatchDataset
from src.models.unmix import UnmixNet
from src.models.cdnet import CDnet
from src.losses import JointLoss
from src.utils.train_utils import load_config, set_seed, save_ckpt, get_device
from src.utils.metrics import compute_metrics
from tqdm import tqdm

def build_dataloaders(cfg, split):
    ds = HSICDPatchDataset(
        root=cfg['data']['root'],
        t1=cfg['data']['t1'],
        t2=cfg['data']['t2'],
        label=cfg['data']['label'],
        mask=cfg['data']['mask'],
        patch=cfg['data']['patch'],
        split=split,
        train_ratio=cfg['data']['train_ratio'],
        max_train_samples=cfg['data']['max_train_samples'],
        balance=cfg['data']['balance'],
        normalize_mode=cfg['data']['normalize'],
        seed=cfg['seed']
    )
    bs = cfg['optim']['batch_size'] if split == 'train' else cfg['eval']['batch_size']
    return ds, DataLoader(ds, batch_size=bs, shuffle=(split == 'train'), num_workers=2, pin_memory=True)

def main(args):
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    device = get_device()

    train_ds, train_loader = build_dataloaders(cfg, 'train')
    test_ds, test_loader = build_dataloaders(cfg, 'test')

    C = cfg['model']['C'] or train_ds.C
    P, K = cfg['model']['P'], cfg['model']['K']
    share_E = cfg['model']['share_E']
    unmix = UnmixNet(C=C, P=P, K=K).to(device)
    cdnet = CDnet(P=P, base=cfg['model']['encoder_ch']).to(device)

    params = list(unmix.parameters()) + list(cdnet.parameters())
    optim = torch.optim.Adam(params, lr=float(cfg['optim']['lr']), weight_decay=float(cfg['optim']['weight_decay']))
    if cfg['optim']['cosine_decay']:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(cfg['optim']['epochs_joint']))
    else:
        sched = None

    
    if args.stage == 'joint' and args.init_from:
        ckpt = torch.load(args.init_from, map_location=device)
        unmix.load_state_dict(ckpt['model']['unmix'])
        
        cdnet.load_state_dict(ckpt['model']['cdnet'], strict=False)
        print(f"Loaded pretrained weights from: {args.init_from}")

    criterion = JointLoss(lambda_rec=cfg['loss']['lambda_rec'],
                          lambda_tv=cfg['loss']['lambda_tv'],
                          lambda_temp=cfg['loss']['lambda_temp'],
                          use_temp=cfg['loss']['use_temp'])

    save_root = os.path.join(cfg['train']['save_dir'], cfg['exp_name'])
    stage_dir = lambda s: os.path.join(save_root, s)
    os.makedirs(save_root, exist_ok=True)
    best_f1 = 0.0

    def run_epoch(loader, train=True, stage='joint', epoch=1, epochs=1):
        nonlocal best_f1
        if train:
            unmix.train(); cdnet.train()
        else:
            unmix.eval(); cdnet.eval()

        losses = []; all_y = []; all_p = []
        pbar = tqdm(loader, ncols=100, desc=f"{'Train' if train else 'Eval'} {stage} [{epoch}/{epochs}]")
        for x1, x2, y in pbar:
            x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)
            (A1, Y1h, _), (A2, Y2h, _) = unmix(x1, x2, share_E=share_E)

            if stage == 'pretrain':
                logits = None
                loss = criterion(logits, y, x1, Y1h, x2, Y2h, A1, A2, None)
            else:
                logits = cdnet(A1, A2)
                loss = criterion(logits, y, x1, Y1h, x2, Y2h, A1, A2, None)

            if train:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(unmix.parameters()) + list(cdnet.parameters()), float(cfg['optim']['grad_clip']))
                optim.step()

            losses.append(loss.item())
            if (stage != 'pretrain') and (logits is not None):
                preds = logits.argmax(1).detach().cpu().numpy().tolist()
                all_p += preds
                all_y += y.detach().cpu().numpy().tolist()

            pbar.set_postfix_str(f"loss={sum(losses)/len(losses):.4f}")

        if stage != 'pretrain':
            metrics = compute_metrics(all_y, all_p)
            
            if (not train) and (metrics['F1'] > best_f1):
                best_f1 = metrics['F1']
                save_ckpt(os.path.join(stage_dir('joint'), 'best.pt'),
                          {'unmix': unmix.state_dict(), 'cdnet': cdnet.state_dict()},
                          optim.state_dict(), epoch=epoch, best=True)
            return sum(losses)/len(losses), metrics
        else:
            return sum(losses)/len(losses), None

    if args.stage == 'pretrain':
        os.makedirs(stage_dir('pretrain'), exist_ok=True)
        epochs_pre = int(cfg['optim']['epochs_pretrain'])
        for ep in range(1, epochs_pre + 1):
            ltr, _ = run_epoch(train_loader, train=True, stage='pretrain', epoch=ep, epochs=epochs_pre)
            lte, _ = run_epoch(test_loader, train=False, stage='pretrain', epoch=ep, epochs=epochs_pre)
            
            print(f"[Epoch {ep}/{epochs_pre}] Train pretrain loss={ltr:.4f}")
            print(f"[Epoch {ep}/{epochs_pre}] Eval  pretrain loss={lte:.4f}")
        save_ckpt(os.path.join(stage_dir('pretrain'), 'last.pt'),
                  {'unmix': unmix.state_dict(), 'cdnet': cdnet.state_dict()},
                  optim.state_dict(), epoch=epochs_pre)
    else:
        os.makedirs(stage_dir('joint'), exist_ok=True)
        epochs_joint = int(cfg['optim']['epochs_joint'])
        for ep in range(1, epochs_joint + 1):
            ltr, mtr = run_epoch(train_loader, train=True, stage='joint', epoch=ep, epochs=epochs_joint)
            lte, mte = run_epoch(test_loader, train=False, stage='joint', epoch=ep, epochs=epochs_joint)
            if sched: sched.step()
            # 每个epoch结束打印一行总览（含指标）
            if mtr is not None:
                print(f"[Epoch {ep}/{epochs_joint}] Train joint loss={ltr:.4f} "
                      f"| OA={mtr['OA']:.4f} F1={mtr['F1']:.4f} P={mtr['Precision']:.4f} R={mtr['Recall']:.4f} Kappa={mtr['Kappa']:.4f}")
            if mte is not None:
                print(f"[Epoch {ep}/{epochs_joint}] Eval  joint loss={lte:.4f} "
                      f"| OA={mte['OA']:.4f} F1={mte['F1']:.4f} P={mte['Precision']:.4f} R={mte['Recall']:.4f} Kappa={mte['Kappa']:.4f}")
        save_ckpt(os.path.join(stage_dir('joint'), 'last.pt'),
                  {'unmix': unmix.state_dict(), 'cdnet': cdnet.state_dict()},
                  optim.state_dict(), epoch=epochs_joint)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--stage", type=str, choices=["pretrain","joint"], required=True)
    ap.add_argument("--init_from", type=str, default="", help="path to pretrain checkpoint (optional)")
    args = ap.parse_args()
    main(args)
