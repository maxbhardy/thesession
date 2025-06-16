CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS Tunes (
    TuneID INTEGER PRIMARY KEY,
    TuneTitle VARCHAR(200),
    TuneAuthor VARCHAR(200),
    TuneURL VARCHAR(200) UNIQUE,
    TuneType VARCHAR(200),
    Tunebooks INTEGER
);

CREATE TABLE IF NOT EXISTS TuneAliases (
    TuneAliasID INTEGER PRIMARY KEY,
    TuneID INTEGER REFERENCES Tunes (TuneID) ON DELETE CASCADE,
    TuneAlias VARCHAR(200)
);

CREATE TABLE IF NOT EXISTS TuneVersions (
    TuneVersionID INTEGER PRIMARY KEY,
    TuneID INTEGER REFERENCES Tunes (TuneID) ON DELETE CASCADE,
    TuneVersion VARCHAR(20000),
    TuneVersionEmbedding VECTOR(512)
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

DROP VIEW IF EXISTS TuneVersionView;
CREATE VIEW TuneVersionView AS
SELECT
    TuneVersionID,
    TuneID,
    ROW_NUMBER() OVER (PARTITION BY TuneID ORDER BY TuneVersionID) AS TuneVersionNumber,
    TuneVersion,
    TuneVersionEmbedding
FROM TuneVersions;