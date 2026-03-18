import os
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def extract_all_features(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 128,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract backbone features, labels, and indices for the full dataset."""
    model.eval()
    features, labels, indices = [], [], []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    idx_offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Assumes backbone returns (logits, features, _)
            _, feat, _ = model(x)
            feat = feat.cpu().numpy()
            features.append(feat)
            labels.append(y.numpy())
            indices.extend(range(idx_offset, idx_offset + x.size(0)))
            idx_offset += x.size(0)

    return np.concatenate(features), np.concatenate(labels), np.array(indices)


def plot_tsne_labeled_and_subset(
    features: np.ndarray,
    labels: np.ndarray,
    all_indices: np.ndarray,
    labeled_indices: Iterable[int],
    subset_indices: Iterable[int],
    selected_indices: Iterable[int],
    cycle: int,
    method: str,
    save_dir: str = "assets/tsne",
) -> str:
    """Plot t-SNE of features and highlight selected samples from the subset."""
    all_indices = np.array(all_indices)
    labeled_indices = np.array(list(labeled_indices))
    subset_indices = np.array(list(subset_indices))
    selected_indices = np.array(list(selected_indices))

    # Flatten and normalize features
    flat_features = np.array([f.flatten() for f in features])
    flat_features = StandardScaler().fit_transform(flat_features)

    # PCA + t-SNE
    pca = PCA(n_components=50, random_state=0).fit_transform(flat_features)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=3000,
        init="pca",
        random_state=0,
    ).fit_transform(pca)

    cifar10_classes = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
    palette = sns.color_palette("tab10", len(cifar10_classes))

    plt.figure(figsize=(10, 8))
    for cls in np.unique(labels):
        cls_mask = labels == cls
        plt.scatter(
            tsne[cls_mask, 0],
            tsne[cls_mask, 1],
            label=cifar10_classes[cls],
            alpha=0.7,
            s=20,
            color=palette[cls],
        )

    selected_mask = np.isin(all_indices, selected_indices)
    plt.scatter(
        tsne[selected_mask, 0],
        tsne[selected_mask, 1],
        c="black",
        marker="x",
        s=20,
        label="Selected",
    )

    plt.xticks([])
    plt.yticks([])
    ax = plt.gca()
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title(f"t-SNE of Labeled + Subset (Cycle {cycle + 1} - {method})", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Class")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"tsne_cycle_{cycle + 1}_{method}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path

