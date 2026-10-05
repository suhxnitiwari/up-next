"""
ENRICH: turn a list of "Watched at" timestamps into sessions and watch time.

Takeout records when a video STARTED, never how long it played. Two derived columns:
  session  - a new session starts after SESSION_GAP_MIN quiet minutes
  dwell_s  - seconds until the next video in the same session, capped at DWELL_CAP_MIN;
             the last video of a session has no successor, so it gets the session's median
             dwell (a guess, flagged by dwell_imputed)
A dwell under 60 s means she moved on almost immediately: a skip, or a Short.
"""
import pandas as pd

import config


def main():
    w = pd.read_parquet(config.DATA / "watches.parquet")
    w = w[w.kind == "video"].sort_values("ts").reset_index(drop=True)

    gap = w.ts.diff().dt.total_seconds()
    w["session"] = (gap.isna() | (gap > config.SESSION_GAP_MIN * 60)).cumsum()

    nxt = w.ts.shift(-1)
    same = w.session.shift(-1) == w.session
    dwell = (nxt - w.ts).dt.total_seconds().where(same)
    w["dwell_imputed"] = dwell.isna()
    dwell = dwell.clip(upper=config.DWELL_CAP_MIN * 60)
    w["dwell_s"] = dwell.fillna(dwell.groupby(w.session).transform("median")).fillna(dwell.median())
    w["skipped"] = ~w.dwell_imputed & (w.dwell_s < 60)

    w["hour"] = w.ts.dt.hour
    w["weekday"] = w.ts.dt.day_name()
    # a 1 AM video belongs to the night it started in, not the morning after
    w["night_of"] = (w.ts - pd.Timedelta(hours=config.NIGHT_ROLLOVER_HOUR)).dt.date
    w["nth_view"] = w.groupby("video_id").cumcount() + 1

    w.to_parquet(config.DATA / "watches_enriched.parquet", index=False)

    s = w.groupby("session").agg(start=("ts", "min"), end=("ts", "max"), videos=("ts", "size"),
                                 minutes=("dwell_s", lambda x: x.sum() / 60))
    s.to_parquet(config.DATA / "sessions.parquet")
    print(f"{len(w):,} videos in {len(s):,} sessions; median session {s.videos.median():.0f} videos, "
          f"{s.minutes.median():.0f} min; {w.skipped.mean():.0%} left within a minute")


if __name__ == "__main__":
    main()
