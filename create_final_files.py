import sqlite3
import pathlib
import re

import music21
import func_timeout
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


# Communication MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

# Only the main node query the database
if mpi_rank == 0:
    query = """
    SELECT TuneVersionID, TuneTitle, TuneVersion
    FROM TuneVersions
    JOIN Tunes USING (TuneID)
    WHERE TuneVersionID != 20508
    ORDER BY TuneVersionID
    """

    # Fetch tunes
    with sqlite3.connect("database2.db") as con:
        tunes = pd.read_sql(query, con)

    con.close()
else:
    tunes = None

# Send tunes and PRNG seed to all nodes
tunes = mpi_comm.bcast(tunes, root=0)

# Select subset
indices = np.array_split(np.arange(len(tunes)), mpi_size)[mpi_rank]
tunes = tunes.iloc[indices].copy()

# Create audio files
root = pathlib.Path("audio_final")
root.mkdir(exist_ok=True)

for row in tunes.itertuples():
    # Create destination directory
    filename = f"{row.TuneVersionID}"

    if not (root / filename).with_suffix(".flac").exists():
        try:
            converter = ABCMusicConverter(row.TuneVersion, filename, root)

            func_timeout.func_timeout(
                30,
                converter.to_flac,
                kwargs=dict(
                    instrument="piano",
                    tempo=180,
                    max_notes=300,
                    cut_silence=30,
                    start=0,
                    duration=60,  # 1 minute
                    sampling_rate=16000,
                    audio_channels=1,
                    clean_files=True,
                ),
            )
            print(row.TuneVersionID, row.TuneTitle)
        except:
            pass
