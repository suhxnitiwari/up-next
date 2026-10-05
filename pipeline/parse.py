"""
PARSE: Google Takeout's YouTube export → tidy parquet tables in data/.

watch-history.html is one 2 MB line of nested <div>s, one "outer-cell" per event.
Each cell has a verb (Watched / Viewed / Used ...), up to two links (video, channel),
a timestamp with a CDT/CST label, and sometimes a "From Google Ads" footer.
Every cell is accounted for: parsed, or counted under the reason it was dropped.
"""
import csv
import html
import json
import re
from collections import Counter
from datetime import datetime

import pandas as pd

import config

CELL = re.compile(r'<div class="outer-cell.*?(?=<div class="outer-cell|\Z)', re.S)
BODY = re.compile(r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>', re.S)
LINK = re.compile(r'<a href="([^"]+)">(.*?)</a>', re.S)
WHEN = re.compile(r'([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}\s[AP]M)\s(C[DS]T)')
VIDEO_ID = re.compile(r'(?:watch\?v=|youtu\.be/|shorts/)([\w-]{11})')
OFFSET = {"CDT": "-05:00", "CST": "-06:00"}


def stamp(text):
    m = WHEN.search(text)
    if not m:
        return None
    local = datetime.strptime(m.group(1).replace(" ", " "), "%b %d, %Y, %I:%M:%S %p")
    return pd.Timestamp(local.isoformat() + OFFSET[m.group(2)]).tz_convert(config.TZ)


def clean(s):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", s)).split())


def cells(path):
    for cell in CELL.findall(path.read_text(encoding="utf-8")):
        body = BODY.search(cell)
        yield cell, (body.group(1) if body else "")


def watch_history():
    rows, dropped = [], Counter()
    for cell, body in cells(config.WATCH_HTML):
        verb = clean(body.split("<a", 1)[0].split("<br>", 1)[0]).split(" ")[0]
        ts = stamp(body)
        if "From Google Ads" in cell:
            dropped["ad"] += 1
            continue
        if verb not in ("Watched", "Viewed") or ts is None:
            dropped[f"not a watch ({verb or 'blank'})"] += 1
            continue
        links = LINK.findall(body)
        vid = next((VIDEO_ID.search(u).group(1) for u, _ in links if VIDEO_ID.search(u)), None)
        channel = next(((u.rsplit("/", 1)[-1], clean(t)) for u, t in links if "/channel/" in u), (None, None))
        if verb == "Watched" and vid is None:
            dropped["watch without a video link"] += 1
            continue
        title = clean(links[0][1]) if links else clean(body.split("<br>", 1)[0])[len(verb):].strip()
        if title.startswith("https://"):
            title = None                    # the video was deleted; YouTube keeps only its URL
        rows.append({
            "ts": ts,
            "kind": "video" if verb == "Watched" else "post",
            "video_id": vid,
            "title": title,
            "channel_id": channel[0],
            "channel": channel[1],
            "clicked_product": "Clicked on product" in body,
        })
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df, dropped


def search_history():
    rows, dropped = [], Counter()
    for cell, body in cells(config.SEARCH_HTML):
        if "From Google Ads" in cell:
            dropped["ad"] += 1
            continue
        if not body.startswith("Searched for"):
            dropped["not a search"] += 1
            continue
        links = LINK.findall(body)
        ts = stamp(body)
        if not links or ts is None:
            dropped["unreadable"] += 1
            continue
        rows.append({"ts": ts, "query": clean(links[0][1])})
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True), dropped


def subscriptions():
    with config.SUBSCRIPTIONS.open() as f:
        return pd.DataFrame([{"channel_id": r["Channel Id"], "channel": " ".join(r["Channel Title"].split())}
                             for r in csv.DictReader(f)])


def playlists():
    with (config.PLAYLISTS / "playlists.csv").open() as f:
        meta = {r["Playlist Title (Original)"]: r for r in csv.DictReader(f)}
    rows = []
    for name in meta:
        path = config.PLAYLISTS / f"{name}-videos.csv"
        if not path.exists():
            continue
        with path.open() as f:
            for r in csv.DictReader(f):
                if r.get("Video ID"):
                    rows.append({"playlist": name, "video_id": r["Video ID"].strip(),
                                 "added": pd.Timestamp(r["Playlist Video Creation Timestamp"]).tz_convert(config.TZ)})
    return pd.DataFrame(rows)


def comments():
    with config.COMMENTS.open() as f:
        return pd.DataFrame([{
            "ts": pd.Timestamp(r["Comment Create Timestamp"]).tz_convert(config.TZ),
            "video_id": r["Video ID"],
            "channel_id": r["Channel ID"],
            "text": " ".join(seg.get("text", "") for seg in json.loads(f"[{r['Comment Text']}]")),
        } for r in csv.DictReader(f)])


def main():
    config.DATA.mkdir(exist_ok=True)
    watches, wdrop = watch_history()
    searches, sdrop = search_history()
    tables = {"watches": watches, "searches": searches, "subscriptions": subscriptions(),
              "playlists": playlists(), "comments": comments()}
    for name, df in tables.items():
        df.to_parquet(config.DATA / f"{name}.parquet", index=False)
        print(f"{name:14} {len(df):6,} rows")
    print("watch cells dropped:", dict(wdrop))
    print("search cells dropped:", dict(sdrop))
    (config.DATA / "parse_log.json").write_text(json.dumps(
        {"watch_dropped": wdrop, "search_dropped": sdrop,
         "rows": {k: len(v) for k, v in tables.items()}}, indent=2))


if __name__ == "__main__":
    main()
