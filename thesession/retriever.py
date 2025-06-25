import pathlib
import psycopg

import numpy as np
import torch
import torchaudio
import pandas as pd

from .model import TheSessionModel


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
