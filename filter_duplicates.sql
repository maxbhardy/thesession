-- Delete tunes with no Versions
DELETE FROM Tunes
WHERE TuneID IN (SELECT TuneID FROM TuneView WHERE NumVersions = 0);

-- Delete duplicates
WITH uniques AS (
    SELECT
        min(TuneID) AS TuneID,
        TuneTitle,
        TuneAuthor,
        TuneType,
        max(Tunebooks) AS Tunebooks,
        NumAliases,
        NumVersions,
        count(TuneID) AS NumDuplicates
    FROM TuneView
    GROUP BY TuneTitle, TuneType, NumAliases, NumVersions
),
duplicates AS (
    SELECT TuneView.TuneID AS TuneID
    FROM TuneView
    LEFT JOIN uniques
        USING (TuneTitle, TuneType, NumAliases, NumVersions)
    WHERE TuneView.TuneID != uniques.TuneID
)
DELETE FROM Tunes WHERE TuneID IN (SELECT TuneID FROM duplicates);

-- Delete Tune versions that are no longer attached to tunes
DELETE FROM TuneVersions WHERE TuneID NOT IN (SELECT TuneID FROM Tunes);

-- Delete Tune aliases that are no longer attached to tunes
DELETE FROM TuneAliases WHERE TuneID NOT IN (SELECT TuneID FROM Tunes);

-- Cleaning DB
VACUUM;