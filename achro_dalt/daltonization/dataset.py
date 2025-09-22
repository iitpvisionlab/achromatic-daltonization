from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from pathlib import Path
from typing import List


class PadDataset(Dataset):
    def __init__(self, image_paths: List[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        image = Image.open(path)
        image = self.transform(image)
        assert image.ndim == 3
        return (image, path.stem)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return TF.to_tensor(image)


def collate_fn(batch) -> tuple[torch.Tensor, List[torch.Size], tuple[str]]:
    images, paths = zip(*batch)

    max_h = max([image.shape[1] for image in images])
    max_w = max([image.shape[2] for image in images])

    padded_images = []
    sizes = []

    for image in images:
        _, h, w = image.shape
        pad_h = max_h - h
        pad_w = max_w - w

        sizes.append(image.shape)
        padding = (0, pad_w, 0, pad_h)
        padding_image = F.pad(image, padding, value=0)
        padded_images.append(padding_image)

    batch_tensor = torch.stack(padded_images)

    return batch_tensor, sizes, paths


def create_dataloader(input_path: Path, batch_size: int):
    files = [
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpeg", ".jpg", ".png"}
    ]
    dataset = PadDataset(sorted(files))
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_fn,
    )


def save_image(image: torch.Tensor, path: Path) -> Path:
    PIL_image = TF.to_pil_image(image)
    PIL_image.save(path, format="PNG")
    return path


def save_results(
    batch: torch.Tensor,
    sizes: torch.Tensor,
    names: list,
    output: Path,
    sfx: str,
) -> List[Path]:

    def add_sfx():
        if sfx:
            return f"_{sfx}"
        return ""

    paths = list()
    for image, size, name in zip(batch, sizes, names):
        _, h, w = size
        path = save_image(image[:, :h, :w], output / f"{name}{add_sfx()}.png")
        paths.append(path)
    return paths
