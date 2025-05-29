PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS Tunes (
    TuneID INTEGER PRIMARY KEY,
    TuneTitle TEXT,
    TuneURL TEXT UNIQUE,
    TuneType TEXT,
    Tunebooks INTEGER
);

CREATE TABLE IF NOT EXISTS TuneAliases (
    TuneAliasID INTEGER PRIMARY KEY,
    TuneID INTEGER REFERENCES Tunes (TuneID),
    TuneAlias TEXT
);

CREATE TABLE IF NOT EXISTS TuneVersions (
    TuneVersionID INTEGER PRIMARY KEY,
    TuneID INTEGER REFERENCES Tunes (TuneID),
    TuneVersion TEXT
);

CREATE VIEW IF NOT EXISTS TuneView AS
SELECT TuneID, TuneTitle, TuneType, Tunebooks, NumAliases, NumVersions
FROM Tunes
LEFT JOIN
    (SELECT TuneID, COUNT(TuneAlias) AS NumAliases FROM TuneAliases GROUP BY TuneID)
USING (TuneID)
LEFT JOIN
    (SELECT TuneID, COUNT(TuneVersion) AS NumVersions FROM TuneVersions GROUP BY TuneID)
USING (TuneID);