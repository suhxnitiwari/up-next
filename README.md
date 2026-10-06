# Watch Next

**Live:** https://suhxnitiwari.github.io/up-next/

Nine months of my YouTube (January–October 2026), parsed from Google Takeout: what I watch, when I watch it, how fast I move on, and which channels I keep coming back to. Some plays (1,061 of 2,037) are left out of the site by choice.

The site is a YouTube watch page built from my own history. Each chapter in the "Up next" queue plays one question in the player, and the filters under it query every play in the browser.

## By the numbers

| Stage | What happens | Scale |
| --- | --- | --- |
| Ingestion | Regex parse of Takeout's single-line HTML; every activity cell is parsed or counted as dropped | 2,060 cells → 2,043 rows (14 ads, 3 non-watch entries dropped) |
| Storage | Tidy pandas tables saved as parquet, kept local | watches, searches, subscriptions, playlists, comments |
| Sessionization | A 30-minute gap starts a new session; watch time = gap to the next start, capped at 30 min | 304 sessions, ~64 estimated hours (after exclusions) |
| Topics | Keyword rules → channel majority → logistic regression on multilingual MiniLM title embeddings | 87% agreement on held-out titles |
| Behavior | Session gateways, topic-to-topic transitions, channel concentration, search-to-click timing | 73 of 509 channels = half of all plays |
| Export | Privacy filters, aggregates and a per-play table for client-side filtering | 976 plays, 582 searchable titles |

## Pipeline

```
Takeout/YouTube and YouTube Music/   Google's HTML + CSV export (not in this repo)
        │
pipeline/parse.py      watch + search history HTML → tidy parquet tables; every cell is parsed or counted as dropped
pipeline/enrich.py     sessions (30 quiet minutes ends one), estimated watch time, skips, "night of" dates
pipeline/classify.py   one topic per video: keyword rules → channel majority → logistic regression on title embeddings
pipeline/analyze.py    every number and title the site shows → site/data.js (privacy filters live here)
        │
site/index.html        the page (vanilla JS, hand-built SVG/CSS charts, no build step)
```

Run it:

```bash
./pipeline/run.sh
```

## Things worth knowing

- **No durations.** Takeout records when a video started, never how long it played. Watch time is the gap until the next video in the same session, capped at 30 minutes. It's an estimate, not a measurement.
- **Ads.** 14 watch entries and 5 search entries were ads ("From Google Ads"). They're dropped and counted.
- **Topics.** Rules plus channel majority label 65% of videos. The embedding model agrees with those labels on 87% of held-out titles and labels the rest when it's confident.
- **Privacy.** Raw exports and parquet tables never enter git. Personal filter lists (`off_limits.txt`, `exclusions.json`, `private_topics.json`) live next to the pipeline and stay out of git too; without them the pipeline still runs, it just filters less. Search queries never reach the site, only their timestamps. Titles on a blocked-topic list never get a topic, so they can't reach the site. Everything runs locally, with no paid APIs.
- **Opening footage.** The opening plays live, muted clips of videos from my own history through YouTube's standard embed, a wall per topic. Nothing is re-hosted. Only clips of real people play there: `frames.sh` and `screen.swift` check three frames from inside each video with Apple's Vision framework, on my laptop, and a clip qualifies only if every frame shows a face, has almost no on-screen text and doesn't read as a drawing.
