from collections.abc import Iterable
import pathlib
import torch
import torchaudio
import numpy as np


class TheSessionDataset(torch.utils.data.Dataset):
    root_dir: pathlib.Path
    sampling_rate: int
    prng: np.random.Generator
    records: list[list[pathlib.Path]]
    device: str | None
    backend: str | None

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        sampling_rate: int = 16000,
        prng: np.random.Generator | int | None = None,
        subset: Iterable | None = None,
        device: str | None = None,
        fmt: str = ".mp3",
        backend: str | None = None,
    ):
        self.root_dir = pathlib.Path(root_dir)
        self.sampling_rate = sampling_rate
        self.device = device
        self.backend = backend

        if isinstance(prng, np.random.Generator):
            self.prng = prng
        else:
            self.prng = np.random.default_rng(prng)

        # Change subset to "set" for faster checks
        if subset is not None:
            subset = set(subset)

        # Regroup recordings of the same recordss together
        records = {}

        for file in self.root_dir.glob(f"**/*{fmt}"):
            if (subset is not None) or (file.parent.stem in subset):
                record_id = file.stem.split("_")[0]

                if record_id in records:
                    records[record_id].append(file)
                else:
                    records[record_id] = [file]

        # Keep only recordings that have 2 or more versions
        self.records = list(r for r in records.values() if len(r) > 1)

    def __len__(self) -> int:
        return len(self.records)

    def load_audio(self, filepath: str | pathlib.Path) -> torch.Tensor:
        waveform, sr = torchaudio.load(filepath, backend=self.backend)

        if self.device:
            waveform = waveform.to(self.device)

        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        return waveform.mean(dim=0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Select 2 versions of the record randomly (without replacement)
        selected = self.prng.choice(self.records[idx], size=2, replace=False)

        return self.load_audio(selected[0]), self.load_audio(selected[1])
