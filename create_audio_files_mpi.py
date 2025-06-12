import sqlite3
import pathlib
import re

import music21
import pandas as pd
import numpy as np

from mpi4py import MPI

from thesession.converter import ABCMusicConverter


def sanitize_title(title: str) -> str:
    # Replace spaces with underscores
    title = title.replace(" ", "_")

    # Remove any character that is NOT alphanumeric, underscore, hyphen, or dot
    title = re.sub(r"[^A-Za-z0-9_\-\.]", "", title)

    # Optionally, truncate length to e.g. 100 chars
    return title[:100].lower()


instruments: dict = {
    k.lower(): v for k, v in vars(music21.instrument).items() if hasattr(v, "bestName")
}

instruments = [
    "piano",
    "accordion",
    "violin",
    "acousticguitar",
    "banjo",
    "flute",
]

tempos = [i for i in range(120, 240, 20)]

# Communication MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# Only the main node query the database
if mpi_rank == 0:
    query = """
    SELECT TuneID, TuneVersionID, TuneTitle, TuneVersion
    FROM TuneVersionView
    JOIN Tunes USING (TuneID)
    WHERE TuneVersionNumber < 6
    ORDER BY TuneID, TuneVersionID
    """

    # Fetch tunes
    with sqlite3.connect("database2.db") as con:
        tunes = pd.read_sql(query, con)

    con.close()

    # Get prng seed
    seed = np.random.default_rng().bit_generator.seed_seq
else:
    tunes = None
    seed = None

# Send tunes and PRNG seed to all nodes
tunes = mpi_comm.bcast(tunes, root=0)
seed = mpi_comm.bcast(seed, root=0)

# Select subset
indices = np.array_split(np.arange(len(tunes)), mpi_size)[mpi_rank]
tunes = tunes.iloc[indices].copy()

# Random generator (spawn from main seed)
prng = np.random.default_rng(seed).spawn(mpi_size)[mpi_rank]
num_audio = 5

# Create audio files
root = pathlib.Path("audio_flac")
root.mkdir(exist_ok=True)

for row in tunes.itertuples():
    # Create destination directory
    title = sanitize_title(row.TuneTitle)
    dest = root / f"{row.TuneID}_{title}"
    dest.mkdir(exist_ok=True)

    # Define the selected instrument and tempo for the tune
    tmp_instruments = prng.choice(instruments, num_audio, replace=False, shuffle=True)
    tmp_tempos = prng.integers(120, 240, size=num_audio)
    tmp_starts = prng.uniform(0, 1, size=num_audio)
    tmp_noises = prng.uniform(0, 0.002, size=num_audio)

    for i, (instr, t, s, n) in enumerate(
        zip(tmp_instruments, tmp_tempos, tmp_starts, tmp_noises)
    ):
        filename = f"{row.TuneVersionID}_{i}"

        if not (dest / filename).with_suffix(".flac").exists():
            try:
                ABCMusicConverter(row.TuneVersion, filename, dest, prng).to_flac(
                    instrument=instr,
                    tempo=t,
                    max_notes=300,
                    cut_silence=30,
                    start=s,
                    duration=60,  # 1 minute
                    noise_amplitude=n,
                    sampling_rate=16000,
                    audio_channels=1,
                    clean_files=True,
                )
                print(row.TuneID, row.TuneTitle, row.TuneVersionID)
            except:
                pass
