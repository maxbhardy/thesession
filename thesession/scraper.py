import numpy as np
import time
import pathlib
import inspect
import shutil
import re
import playwright.sync_api
import bs4
import sqlite3
from typing import Any
from collections.abc import Callable, Iterator, Iterable


class Scraper:
    def __init__(self, crawl_delay=(60, 15), headless=True):
        self.crawl_delay = crawl_delay
        self.headless = headless
        self.browser = None
        self.context = None

    def start(self) -> None:
        self.playwright = playwright.sync_api.sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        )

    def close(self) -> None:
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def sleep_crawl_delay(self) -> None:
        # Wait a duration distributed normally around mean with standard deviation std
        time.sleep(np.random.normal(*self.crawl_delay))

    def fetch_page(self, url: str, on_result=None, **kwargs) -> str:
        page = self.context.new_page()
        page.goto(url)
        html = page.content()
        page.close()

        if on_result:
            return on_result(url, html, **kwargs)
        else:
            return html

    def fetch_pages(
        self,
        urls: Iterable[str],
        on_result: Callable[[str, str], Any] | None = None,
        **kwargs,
    ) -> dict:
        res = {}

        for i, url in enumerate(urls):
            # Determine whether this url should be fetched

            # Sleep the crawl delay
            if i != 0:
                self.sleep_crawl_delay()

            res[url] = self.fetch_page(url, on_result, **kwargs)

        return res

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


def extract_metadata(soup: bs4.BeautifulSoup) -> dict:
    # Extract main title from <h1>
    h1_tag = soup.find("h1")
    title = h1_tag.contents[0].strip() if h1_tag else None

    # Extract tune type from <h1 <span/>/>
    h1_tag_span = h1_tag.find("span", class_="detail")
    tunetype = h1_tag_span.get_text() if h1_tag_span else None

    # Extract tune author
    by_tag = soup.find("p", class_="manifest-item-title")
    author = by_tag.get_text().strip().lstrip("By").strip() if by_tag else None

    # Extract "Also known as" names from <p class="info">
    info_p = soup.find("p", class_="info")
    aliases = []
    if info_p and info_p.text.startswith("Also known as"):
        raw = info_p.text.replace("Also known as", "").strip()
        aliases = [name.strip() for name in raw.split(",")]

    # Extract number of tunebooks
    tunebooks = soup.find("div", {"id": "tunebooking"})
    if tunebooks:
        try:
            tunebooks = int(
                tunebooks.get_text().strip().split()[-2].strip().replace(",", "")
            )
        except:
            tunebooks = None

    return {
        "title": title,
        "author": author,
        "type": tunetype,
        "tunebooks": tunebooks,
        "aliases": aliases,
    }


def extract_abc_notation(soup: bs4.BeautifulSoup, abc_id: str = "abc1") -> str:
    """
    Extracts ABC music notation from a specific <div> block in given HTML.

    Parameters:
        html (str): The full HTML content as a string.
        abc_id (str): The ID of the ABC notation block to extract (default: "abc7").

    Returns:
        str: The extracted ABC notation as a plain text string.
    """
    abc_div = soup.find("div", {"class": "setting-abc", "id": abc_id})
    if not abc_div:
        raise ValueError(f"No div found with id '{abc_id}'")

    notes_div = abc_div.find("div", {"class": f"notes"})
    if not notes_div:
        raise ValueError(f"No notes div found")

    # Extract the text, convert HTML entities, and strip leading/trailing whitespace
    abc_text = notes_div.get_text()
    return abc_text.strip()


def extract_abc_versions(soup: bs4.BeautifulSoup) -> list[str]:
    i = 1
    versions = []

    while True:
        try:
            abc = extract_abc_notation(soup, abc_id=f"abc{i}")
            versions.append(abc)
            i += 1
        except ValueError:
            break

    return versions


def sanitize_title(title: str) -> str:
    # Replace spaces with underscores
    title = title.replace(" ", "_")

    # Remove any character that is NOT alphanumeric, underscore, hyphen, or dot
    title = re.sub(r"[^A-Za-z0-9_\-\.]", "", title)

    # Optionally, truncate length to e.g. 100 chars
    return title[:100].lower()


def parse_page(
    url: str,
    html: str,
    database: str | pathlib.Path,
    base_url: str,
    verbose: bool = False,
) -> None:
    # Create beautiful soup
    soup = bs4.BeautifulSoup(html, "html.parser")

    # Fetch title, aliases and ABC versions
    results = extract_metadata(soup)

    results["url"] = url
    results["number"] = int(url.lstrip(base_url).lstrip("/"))

    # Fetch ABC notations of the different versions
    results["versions"] = extract_abc_versions(soup)

    if verbose:
        print(f"{results['number']} - {results['title']}")

    # Write to database
    if results["title"] not in [404, "Forbidden", "404", 410, "410"]:
        with sqlite3.connect(database) as con:
            tune_id = results["number"]

            # Insert tune
            con.execute(
                "INSERT OR REPLACE INTO Tunes "
                "(TuneID, TuneTitle, TuneAuthor, TuneURL, TuneType, Tunebooks) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tune_id,
                    results["title"],
                    results["author"],
                    results["url"],
                    results["type"],
                    results["tunebooks"],
                ),
            )

            # Delete previous aliases and versions
            con.execute("DELETE FROM TuneAliases WHERE TuneID = ?", (tune_id,))
            con.execute("DELETE FROM TuneVersions WHERE TuneID = ?", (tune_id,))

            # Insert Tune aliases
            con.executemany(
                "INSERT INTO TuneAliases (TuneID, TuneAlias) VALUES (?, ?)",
                [(tune_id, a) for a in results["aliases"]],
            )

            con.executemany(
                "INSERT INTO TuneVersions (TuneID, TuneVersion) VALUES (?, ?)",
                [(tune_id, v) for v in results["versions"]],
            )

            con.commit()
        con.close()


def url_generator(base_url: str, elements: Iterable[int]) -> Iterator[str]:
    # Add trailing slash
    if base_url[-1] != "/":
        base_url = base_url + "/"

    for i in elements:
        yield base_url + f"{i}"


def create_database(
    database: str | pathlib.Path, schema: str | pathlib.Path, overwrite: bool = False
) -> None:
    database = pathlib.Path(database)
    schema = pathlib.Path(schema)

    if database.exists() and overwrite:
        database.unlink()

    with sqlite3.connect(database) as con:
        # Create schema
        with open(schema, "r") as f:
            con.executescript(f.read())

    con.close()


if __name__ == "__main__":
    # Create database
    create_database("database.db", "database.sql", overwrite=False)

    # Random generator (with fixed seed)
    prng = np.random.default_rng(4567890123)

    # Tunes
    tunes = prng.permutation(np.arange(1, 25000))

    # Find tunes present in database
    with sqlite3.connect("database.db") as con:
        cursor = con.execute("SELECT TuneID FROM Tunes")
        present = cursor.fetchall()
        present = np.array([r[0] for r in present], dtype=int)

    con.close()

    # Return tunes that are not present in database
    tunes = np.setdiff1d(tunes, present, assume_unique=True)

    # Start scraping
    base_url = "https://thesession.org/tunes/"
    urls = url_generator(base_url, tunes)

    with Scraper() as s:
        s.fetch_pages(
            urls,
            on_result=parse_page,
            database="database.db",
            base_url=base_url,
            verbose=True,
        )
