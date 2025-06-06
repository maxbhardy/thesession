PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS Tunes (
    TuneID INTEGER PRIMARY KEY,
    TuneTitle TEXT,
    TuneAuthor TEXT,
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

DROP VIEW IF EXISTS TuneView;
CREATE VIEW TuneView AS
SELECT
    TuneID,
    TuneTitle,
    TuneAuthor,
    TuneType,
    Tunebooks,
    coalesce(NumAliases, 0) AS NumAliases,
    coalesce(NumVersions, 0) AS NumVersions
FROM Tunes
LEFT JOIN
    (SELECT TuneID, count(TuneAlias) AS NumAliases FROM TuneAliases GROUP BY TuneID)
USING (TuneID)
LEFT JOIN
    (SELECT TuneID, count(TuneVersion) AS NumVersions FROM TuneVersions GROUP BY TuneID)
USING (TuneID);