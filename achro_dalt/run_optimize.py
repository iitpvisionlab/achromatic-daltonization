import argparse
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np
import numpy.typing as npt
import json
from datetime import datetime

from .config import Config
from .daltonization.optimize import optimize
from .evaluation.metrics import CD, RMS


class Metrics:
    cd_lab = CD("lab")
    cd_prolab = CD("prolab")
    rms_lab = RMS("lab")
    rms_prolab = RMS("prolab")

    def __init__(self) -> None:
        self.cd_full_color_lab = list()
        self.cd_full_color_prolab = list()
        self.cd_sim_lab = list()
        self.cd_sim_prolab = list()
        self.rms_contrast_lab = list()
        self.rms_contrast_prolab = list()
        self._cnt = 0

    def calc_CD(self, path1: Path, path2: Path) -> tuple[float, float]:
        image1 = read_image(path1)
        image2 = read_image(path2)
        return self.cd_lab(image1, image2), self.cd_prolab(image1, image1)

    def calc_RMS(self, path1: Path, path2: Path) -> tuple[float, float]:
        image1 = read_image(path1)
        image2 = read_image(path2)
        return self.rms_lab(image1, image2), self.rms_prolab(image1, image2)

    def add(self, orig: Path, dalt: Path, orig_sim: Path, dalt_sim: Path):
        lab, prolab = self.calc_CD(orig, dalt)
        self.cd_full_color_lab.append(lab)
        self.cd_full_color_prolab.append(prolab)

        lab, prolab = self.calc_CD(orig_sim, dalt_sim)
        self.cd_sim_lab.append(lab)
        self.cd_sim_prolab.append(prolab)

        lab, prolab = self.calc_RMS(orig, dalt_sim)
        self.rms_contrast_lab.append(lab)
        self.rms_contrast_prolab.append(prolab)

        self._cnt += 1
        return self._as_dict(
            self.cd_full_color_lab[-1],
            self.cd_full_color_prolab[-1],
            self.cd_sim_lab[-1],
            self.cd_sim_prolab[-1],
            self.rms_contrast_lab[-1],
            self.rms_contrast_prolab[-1],
        )

    def summary(self):
        assert self._cnt > 0, "Not enough data"
        values = list(
            map(
                lambda x: round(np.mean(x), 4),
                [
                    self.cd_full_color_lab,
                    self.cd_full_color_prolab,
                    self.cd_sim_lab,
                    self.cd_sim_prolab,
                    self.rms_contrast_lab,
                    self.rms_contrast_prolab,
                ],
            )
        )
        return self._as_dict(*values)

    def _as_dict(
        self,
        cd_full_color_lab: float,
        cd_full_color_prolab: float,
        cd_sim_lab: float,
        cd_sim_prolab: float,
        rms_contrast_lab: float,
        rms_contrast_prolab: float,
    ):
        return {
            "cd_full_color_lab": cd_full_color_lab,
            "cd_full_color_prolab": cd_full_color_prolab,
            "cd_sim_lab": cd_sim_lab,
            "cd_sim_prolab": cd_sim_prolab,
            "rms_contrast_lab": rms_contrast_lab,
            "rms_contrast_prolab": rms_contrast_prolab,
        }


def read_image(path: Path) -> npt.NDArray:
    return np.asarray(Image.open(path), dtype=np.float32) / 255.0


def generate_path_set(
    paths: List[Path],
    dataset_path: Path,
    dalt_sfx: str = "dalt",
    orig_sim_sfx: str = "orig_sim",
    dalt_sim_sfx: str = "dalt_sim",
) -> List[tuple[Path, Path, Path, Path]]:
    items = list()
    for p in paths:
        basename = p.name.rpartition(f"_{dalt_sfx}")[0]
        orig_path = list(dataset_path.glob(f"{basename}.*"))
        if len(orig_path) > 0:
            orig_path = orig_path[0]
        else:
            print(f"File with pattern {dataset_path / basename}.* not found")
            continue
        orig_sim = p.parent / f"{basename}_{orig_sim_sfx}.png"
        if not orig_sim.is_file():
            print(f"File {orig_sim} not found")
            continue
        dalt_sim = p.parent / f"{basename}_{dalt_sim_sfx}.png"
        if not dalt_sim.is_file():
            print(f"File {dalt_sim} not found")
            continue

        items.append((orig_path, p, orig_sim, dalt_sim))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()

    config = Config.read_config(args.config)
    device = config.cuda
    sim_func = config.simulation.create(device)
    create_optimizer = config.optimizer.load()
    create_loss = config.loss.create(sim_func)

    cvd_type = config.simulation.cvd_type
    sim_type = config.simulation.name
    problem_name = config.loss.name
    now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    save_dir = config.save_dir / "_".join(
        [now, problem_name, sim_type, cvd_type]
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    dalt_paths = optimize(
        sim_func,
        create_optimizer,
        create_loss,
        config.epochs,
        config.gap,
        config.dataset_path,
        config.batch_size,
        save_dir,
        config.cuda,
    )

    items = generate_path_set(dalt_paths, config.dataset_path)

    metrics = Metrics()
    for orig, dalt, orig_sim, dalt_sim in items:
        stat = metrics.add(orig, dalt, orig_sim, dalt_sim)
        (save_dir / f"{dalt.name}.json").write_text(json.dumps(stat, indent=4))

    (save_dir / "summary.json").write_text(
        json.dumps(metrics.summary(), indent=4)
    )
    print(save_dir / "summary.json")


if __name__ == "__main__":
    main()
