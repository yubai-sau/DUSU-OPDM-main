import numpy as np
import matplotlib.pyplot as plt

def plot_change_map(prob, label=None, out_png=None):
    plt.figure()
    plt.imshow(prob, vmin=0.0, vmax=1.0)
    plt.title('Change Probability')
    plt.colorbar()
    if out_png: plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close()

    if label is not None:
        plt.figure()
        plt.imshow(label, vmin=0, vmax=1)
        plt.title('Ground Truth')
        plt.colorbar()
        if out_png:
            base = out_png.replace('.png','_gt.png')
            plt.savefig(base, dpi=200, bbox_inches='tight')
        plt.close()


def plot_binary_map(mask, out_png, title="Binary Map (0=black, 1=white)"):
    """保存黑白二值图：0=黑(无变化), 1=白(变化)"""
    plt.figure()
    plt.imshow(mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_side_by_side(img_left, img_right, out_png, title_left="Pred", title_right="GT"):
    """并排对比图（用于显示二值预测 vs 真值）"""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_left.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axs[0].set_title(title_left); axs[0].axis("off")
    axs[1].imshow(img_right.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axs[1].set_title(title_right); axs[1].axis("off")
    fig.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

def plot_confusion_overlay(pred_bin, gt_bin, out_png):
    """
    可选：用颜色显示 TP/TN/FP/FN：
    - TN(0,0)=黑, TP(1,1)=白, FP(1,0)=红, FN(0,1)=蓝
    """
    h, w = gt_bin.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    tn = (pred_bin == 0) & (gt_bin == 0)
    tp = (pred_bin == 1) & (gt_bin == 1)
    fp = (pred_bin == 1) & (gt_bin == 0)
    fn = (pred_bin == 0) & (gt_bin == 1)
    rgb[tn] = [0, 0, 0]       # black
    rgb[tp] = [1, 1, 1]       # white
    rgb[fp] = [1, 0, 0]       # red
    rgb[fn] = [0, 0, 1]       # blue
    plt.figure()
    plt.imshow(rgb)
    plt.title("Confusion Overlay (TN=black, TP=white, FP=red, FN=blue)")
    plt.axis("off")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
