import pathlib
import psycopg
import re
import shutil

import numpy as np
import torch
import torchaudio
import pandas as pd
from collections.abc import Iterable

from .model import TheSessionModel
from .converter import ABCMusicConverter

def sanitize_title(title: str) -> str:
    # Replace spaces with underscores
    title = title.replace(" ", "_")

    # Remove any character that is NOT alphanumeric, underscore, hyphen, or dot
    title = re.sub(r"[^A-Za-z0-9_\-\.]", "", title)

    # Optionally, truncate length to e.g. 100 chars
    return title[:100].lower()


def get_database_url(
    user: str,
    password: str,
    host: str = "localhost",
    port: str | int = 5432,
    database: str | None = None,
):
    if database is None:
        database = user

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


class TheSessionRetriever:
    database_url: str
    model: TheSessionModel
    sampling_rate: int
    backend: str | None
    device: str | None
    query = str

    def __init__(
        self,
        database_url: str,
        model_weights: str | pathlib.Path,
        sampling_rate: int = 48000,
        backend: str | None = None,
        device: str | None = None,
    ):
        self.database_url = database_url
        self.sampling_rate = sampling_rate
        self.backend = backend
        self.device = device

        self.model = TheSessionModel(device=self.device)
        self.model.toggle_gradients(False, verbose=False)
        self.model.load(model_weights)
        self.model.eval()

        self.query = """
        SELECT
            TuneID,
            TuneVersionID,
            TuneTitle,
            TuneAuthor,
            TuneURL,
            TuneType,
            Tunebooks,
            TuneVersion,
            1.0 - (TuneVersionEmbedding <=> %s::VECTOR) AS Similarity
        FROM TuneVersions
        JOIN Tunes USING (TuneID)
        WHERE TuneVersionEmbedding IS NOT NULL
        ORDER BY Similarity DESC
        LIMIT %s
        """

        self.generate_audio_query = """
        SELECT
            TuneID,
            TuneVersionID,
            TuneTitle,
            TuneVersion
        FROM TuneVersions
        JOIN Tunes USING (TuneID)
        WHERE TuneVersionID IN ({placeholders})
        """

    def load_audio(
        self,
        filepath: str | pathlib.Path,
        duration: int | None = None,
        signal_multiplier: int | None = None,
        skip_start: int | None = None,
        backend: str | None = None,
    ) -> torch.Tensor:
        if not backend:
            backend = self.backend

        signal, sr = torchaudio.load(filepath, backend=backend)

        if self.device:
            signal = signal.to(self.device)

        # Resample
        if sr != self.sampling_rate:
            signal = torchaudio.functional.resample(signal, sr, self.sampling_rate)

        # Convert to mono
        signal = signal.mean(dim=0)

        # Multiply by multiplier
        if signal_multiplier:
            signal = signal * signal_multiplier

        if skip_start:
            signal = signal[skip_start * self.sampling_rate :]

        # Cut/repeat to fix duration
        if duration:
            length = len(signal)
            expected_length = int(np.round(duration * self.sampling_rate))
            repeats = int(np.ceil(expected_length / length))
            signal = torch.tile(signal, (repeats,))[:expected_length]

        return signal

    def generate_audio(self, version_ids: int | Iterable[int], destination: str | pathlib.Path = ".", fmt: str = "mp3", replace: bool = False, **kwargs):
        destination = pathlib.Path(destination)

        if replace and destination.exists():
            shutil.rmtree(destination)

        destination.mkdir(exist_ok=True, parents=True)

        if isinstance(version_ids, int):
            version_ids = [version_ids]


        with psycopg.connect(self.database_url) as con:
            query = self.generate_audio_query.format(
                placeholders=", ".join(["%s"] * len(version_ids))
            )
            cursor = con.execute(query, (*version_ids,))
            rows = cursor.fetchall()

        df = pd.DataFrame(
            rows,
            columns=[
                "TuneID",
                "TuneVersionID",
                "TuneTitle",
                "TuneVersion",
            ],
        )

        for row in df.itertuples():
            title = sanitize_title(row.TuneTitle)
            filename = f"{row.TuneID}_{title}_{row.TuneVersionID}"

            converter = ABCMusicConverter(row.TuneVersion, filename, destination)

            if fmt == "midi":
                converter.to_midi(**kwargs)
            elif fmt == "wav":
                converter.to_wav(clean_files=True, **kwargs)
            elif fmt == "mp3":
                converter.to_mp3(clean_files=True, **kwargs)
            elif fmt == "flac":
                converter.to_flac(clean_files=True, **kwargs)
            else:
                raise RuntimeError(f"Unknown format {fmt}")

    def compute_embedding(
        self, data: torch.Tensor, unsqueeze: bool = False
    ) -> torch.Tensor:
        if unsqueeze:
            return self.model(data.unsqueeze(0)).flatten()
        else:
            return self.model(data)

    def __call__(
        self, filepath: str | pathlib.Path, limit: int = 5, **kwargs
    ) -> pd.DataFrame:
        # Load audio into tensor
        filepath = pathlib.Path(filepath)
        signal = self.load_audio(filepath, **kwargs)

        # Compute embedding of audio file
        embedding = self.compute_embedding(signal, unsqueeze=True)

        # Get best matches from database
        with psycopg.connect(self.database_url) as con:
            cursor = con.execute(self.query, (embedding.cpu().tolist(), limit))
            rows = cursor.fetchall()

        df = pd.DataFrame(
            rows,
            columns=[
                "TuneID",
                "TuneVersionID",
                "TuneTitle",
                "TuneAuthor",
                "TuneURL",
                "TuneType",
                "Tunebooks",
                "TuneVersion",
                "Similarity",
            ],
        )

        return df
