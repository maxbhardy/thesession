import os
import sqlite3
import psycopg

import numpy as np
import torch
import dotenv

from thesession.dataset import TheSessionFinalDataset
from thesession.model import TheSessionModel

dotenv.load_dotenv("thesession-db/.env")

def get_database_credentials():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    port = "15432"
    url = "localhost"

    return f"postgresql://{user}:{password}@{url}:{port}/{db}"

sqlite_db = "database2.db"

# Load tunes form sqlite and write them into postgres
with sqlite3.connect(sqlite_db) as con:
    cursor = con.execute(
        "SELECT TuneID, TuneTitle, TuneAuthor, TuneURL, TuneType, Tunebooks FROM Tunes ORDER BY TuneID"
    )

    rows = cursor.fetchall()


with psycopg.connect(get_database_credentials()) as con:
    cursor = con.cursor()
    cursor.executemany(
        "INSERT INTO Tunes (TuneID, TuneTitle, TuneAuthor, TuneURL, TuneType, Tunebooks) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
        rows
    )

    con.commit()

# Load tune aliases from sqlite and write into postgres
with sqlite3.connect(sqlite_db) as con:
    cursor = con.execute(
        "SELECT TuneAliasID, TuneID, TuneAlias FROM TuneAliases ORDER BY TuneAliasID"
    )

    rows = cursor.fetchall()


with psycopg.connect(get_database_credentials()) as con:
    cursor = con.cursor()
    cursor.executemany(
        "INSERT INTO TuneAliases (TuneAliasID, TuneID, TuneAlias) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        rows
    )

    con.commit()


# Load tune aliases from sqlite and write into postgres
with sqlite3.connect(sqlite_db) as con:
    cursor = con.execute(
        "SELECT TuneVersionID, TuneID, TuneVersion FROM TuneVersions ORDER BY TuneVersionID"
    )

    rows = cursor.fetchall()


with psycopg.connect(get_database_credentials()) as con:
    cursor = con.cursor()
    cursor.executemany(
        "INSERT INTO TuneVersions (TuneVersionID, TuneID, TuneVersion) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
        rows
    )

    con.commit()


# Compute tune version embeddings and write into database
dataset = TheSessionFinalDataset(
    "audio_final",
    sampling_rate=48000,
    fmt=".flac",
    backend="soundfile",
)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=False, num_workers=6
)

# Model
model = TheSessionModel()
model.load("models/step6_best.pt")
model.toggle_gradients(False, verbose=False)
model.eval()

for record_ids, records in data_loader:
    embeddings = model(records)

    results = [(embeddings[i].cpu().tolist(), record_ids[i].item()) for i in range(len(record_ids))]

    with psycopg.connect(get_database_credentials()) as con:
        cursor = con.cursor()

        cursor.executemany(
            "UPDATE TuneVersions SET TuneVersionEmbedding = %s WHERE TuneVersionID = %s",
            results
        )

    
