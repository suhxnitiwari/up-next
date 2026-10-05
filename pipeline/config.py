"""Paths and constants shared by the pipeline. Raw data never leaves the Takeout folder."""
from pathlib import Path
from zoneinfo import ZoneInfo

TAKEOUT = Path.home() / "Desktop/Personal/Google Data/Takeout 2/YouTube and YouTube Music"
WATCH_HTML = TAKEOUT / "history/watch-history.html"
SEARCH_HTML = TAKEOUT / "history/search-history.html"
SUBSCRIPTIONS = TAKEOUT / "subscriptions/subscriptions.csv"
PLAYLISTS = TAKEOUT / "playlists"
COMMENTS = TAKEOUT / "comments/comments.csv"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"                    # private parquet tables, never committed
SITE = ROOT / "site"

TZ = ZoneInfo("America/Chicago")        # Austin
SESSION_GAP_MIN = 30                    # 30 quiet minutes ends a session
DWELL_CAP_MIN = 30                      # a gap longer than this means she left, not that she watched
NIGHT_ROLLOVER_HOUR = 5                 # 2 AM still counts as the night before
