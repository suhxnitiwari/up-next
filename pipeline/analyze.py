"""
ANALYZE: every number and title the site shows → site/data.js.

Two filters happen here. Plays matching a rule in the private exclusions.json are left out of the
site entirely, charts and totals included. Then a title reaches the site only if it passed the
off-limits list and still exists (deleted videos have no title). Search queries never leave: only
their timestamps do. Emojis are stripped from every title.
"""
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

import config
from classify import OFF_LIMITS

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍⌀-⏿]")
NO_HEADLINES = set()
# Plays matching any rule in exclusions.json are left out of the site, charts and totals included.
# The rules are personal, so the file sits next to this one and stays out of git, like off_limits.txt.
# A rule matches when every condition it lists holds: topics (null = untagged), channels, and regex patterns.
_EXCL = Path(__file__).resolve().parent / "exclusions.json"
RULES = [{"topics": r.get("topics"), "channels": r.get("channels"),
          "all": [re.compile(p, re.I) for p in r.get("all", [])]}
         for r in (json.loads(_EXCL.read_text())["rules"] if _EXCL.exists() else [])]


def excluded(df, ignore_topic=False):
    """Boolean mask of rows any rule matches. ignore_topic is for rows with no topic yet (Watch later)."""
    text = df.title.fillna("") + " | " + df.channel.fillna("")
    out = pd.Series(False, index=df.index)
    for r in RULES:
        if ignore_topic and not (r["channels"] or r["all"]):
            continue
        m = pd.Series(True, index=df.index)
        if r["topics"] and not ignore_topic:
            named = [t for t in r["topics"] if t]
            m &= df.topic.isin(named) | (df.topic.isna() if None in r["topics"] else False)
        if r["channels"]:
            m &= df.channel.isin(r["channels"])
        for rx in r["all"]:
            m &= text.str.contains(rx)
        out |= m
    return out


# Four topic groups for charts that need categorical color (validated palette, 4 hues + gray).
GROUPS = {"Makeup & hair": ["Makeup & hair"],
          "Self-growth": ["Mindset & discipline", "Femininity & dating", "Calm & wellbeing", "Talks & ideas"],
          "Film & TV": ["Film & TV"], "Food & hobbies": ["Food", "Art & making"]}
GROUP_NAMES = list(GROUPS) + ["Other"]
TOPIC_GROUP = {t: i for i, g in enumerate(GROUPS.values()) for t in g}
TOPIC_ORDER = ["Makeup & hair", "Mindset & discipline",
               "Film & TV", "Femininity & dating", "Calm & wellbeing", "Talks & ideas", "Food", "Art & making"]


def tidy(title):
    return " ".join(EMOJI.sub("", title or "").split()).strip(" |-")


def oembed(ids):
    """Titles for saved videos that never appear in watch history (public oEmbed, cached)."""
    path = config.DATA / "oembed.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    for vid in ids:
        if vid in cache:
            continue
        url = "https://www.youtube.com/oembed?format=json&url=" + quote(f"https://www.youtube.com/watch?v={vid}")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                d = json.load(r)
            cache[vid] = {"title": d.get("title"), "channel": d.get("author_name")}
        except Exception:
            cache[vid] = None               # private or deleted
    path.write_text(json.dumps(cache, indent=1))
    return cache


def card(row):
    return {"id": row.video_id, "t": tidy(row.title), "ch": row.channel or "", "n": int(row.n),
            "topic": row.topic, "first": row.first.strftime("%b %-d"), "last": row.last.strftime("%b %-d"),
            "dw": int(round(row.dwell))}


def main():
    w = pd.read_parquet(config.DATA / "watches_enriched.parquet")
    v = pd.read_parquet(config.DATA / "videos.parquet")
    w = w.merge(v[["video_id", "topic", "off_limits"]], on="video_id", how="left")
    sessions = pd.read_parquet(config.DATA / "sessions.parquet")
    searches = pd.read_parquet(config.DATA / "searches.parquet")
    subs = pd.read_parquet(config.DATA / "subscriptions.parquet")
    playlists = pd.read_parquet(config.DATA / "playlists.parquet")
    log = json.loads((config.DATA / "parse_log.json").read_text())
    clog = (config.DATA / "classify_log.txt").read_text()

    drop = excluded(w)
    n_all, n_excluded = len(w), int(drop.sum())
    # Dwell was measured on the full history in enrich.py, so the gaps around a removed play stay real.
    w_all = w
    w = w[~drop].reset_index(drop=True)
    sessions = (w.groupby("session").agg(start=("ts", "min"), end=("ts", "max"), videos=("ts", "size"),
                                         minutes=("dwell_s", lambda x: x.sum() / 60)))
    showable = (w.title.notna() & ~w.off_limits.fillna(True) & w.topic.notna() & ~w.topic.isin(NO_HEADLINES))
    per_video = (w[showable].groupby("video_id")
                 .agg(title=("title", "first"), channel=("channel", "first"), topic=("topic", "first"),
                      n=("ts", "size"), first=("ts", "min"), last=("ts", "max"), dwell=("dwell_s", "median"))
                 .reset_index())

    # ---- home shelves
    rewatched = per_video[per_video.n >= 3].sort_values(["n", "last"], ascending=False)
    late = w[showable & w.hour.between(0, 4)]
    late_top = (late.groupby("video_id").size().rename("k").reset_index()
                .merge(per_video, on="video_id").sort_values(["k", "n"], ascending=False))
    by_topic = {t: [card(r) for r in per_video[per_video.topic == t].sort_values(["n", "last"], ascending=False)
                    .head(24).itertuples()] for t in TOPIC_ORDER if t not in NO_HEADLINES}

    # ---- when
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grid = pd.crosstab(w.weekday, w.hour).reindex(index=days, columns=range(24), fill_value=0)
    topic_hour = pd.crosstab(w.topic, w.hour).reindex(index=TOPIC_ORDER, columns=range(24), fill_value=0)
    band = pd.cut(w.hour, [-1, 4, 11, 16, 20, 23], labels=["late", "morning", "afternoon", "evening", "night"])
    band_mix = pd.crosstab(w.topic, band, normalize="columns").reindex(TOPIC_ORDER).fillna(0)
    peak_band = {b: band_mix[b].idxmax() for b in band_mix.columns}

    # ---- seasons: topic share by month
    month = w.ts.dt.strftime("%Y-%m")
    tm = pd.crosstab(w.topic, month).reindex(TOPIC_ORDER).fillna(0)
    months = list(tm.columns)

    # ---- attention: how long she stays, by topic
    att = (w[w.topic.notna()].groupby("topic")
           .agg(n=("ts", "size"), skip=("skipped", "mean"), dwell=("dwell_s", "median"))
           .reindex(TOPIC_ORDER))

    # ---- sessions
    sizes = sessions.videos
    size_bins = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 25, "11-25"), (26, 10**6, "26+")]
    biggest = sessions.sort_values("videos", ascending=False).iloc[0]
    big = w[w.session == biggest.name]
    big_topics = big.topic.fillna("Other").value_counts()
    big_strip = [{"id": r.video_id, "t": tidy(r.title), "m": int((r.ts - big.ts.min()).total_seconds() // 60)}
                 for r in big[showable.loc[big.index]].drop_duplicates("video_id").itertuples()][::4][:36]

    # ---- channels
    ch = w[w.channel.notna()].groupby(["channel_id", "channel"]).size().rename("n").reset_index()
    loyal_bins = [(1, 1, "Once"), (2, 4, "2-4"), (5, 9, "5-9"), (10, 10**6, "10+")]
    ch_ok = ch[~ch.channel.str.contains(OFF_LIMITS)]
    bad_ch = set(w.groupby("channel_id").off_limits.mean().pipe(lambda s: s[s > 0.5]).index)
    top_channels = ch_ok[~ch_ok.channel_id.isin(bad_ch)].sort_values("n", ascending=False).head(15)
    ch_topic = w.groupby("channel_id").topic.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    sub_ids = set(subs.channel_id)
    sub_watched = ch[ch.channel_id.isin(sub_ids)]

    # ---- searches: counts only
    se = searches.sort_values("ts").reset_index(drop=True)
    wt = w_all.ts.sort_values().reset_index(drop=True)      # search timing is behavior, so it uses the full history
    pos = np.searchsorted(wt.values, se.ts.values)
    nxt = pd.Series([wt.iloc[p] if p < len(wt) else pd.NaT for p in pos])
    gap_min = (nxt - se.ts).dt.total_seconds() / 60

    # ---- the opening only plays clips of real people: frames screened by screen.swift must each show a face,
    # carry almost no on-screen text, and not read as a drawing or animation
    people = []
    screen_path = config.DATA / "screen.json"
    if screen_path.exists():
        scr = json.loads(screen_path.read_text())
        def real(vid):
            fr = scr.get(vid, {}).get("frames", [])
            return len(fr) == 3 and all(f["face"] >= 0.015 and f["text"] <= 0.02 and f["drawn"] < 0.2 for f in fr)
        people = [vid for vid in per_video.video_id if real(vid)]

    # ---- playlists: every saved video, titled via oEmbed, with the same exclusions as the history
    seen = set(w.video_id)
    meta = oembed(playlists.video_id.tolist())

    def playlist_rows(pl):
        rows = []
        for r in pl.itertuples():
            m = meta.get(r.video_id)
            title = tidy(m["title"]) if m and m.get("title") else None
            ch_name = (m or {}).get("channel") or ""
            row = pd.DataFrame({"title": [title], "channel": [ch_name], "topic": [None]})
            if (title and OFF_LIMITS.search(title + " | " + ch_name)) or excluded(row, ignore_topic=True).iat[0]:
                continue
            rows.append({"id": r.video_id, "t": title, "ch": ch_name,
                         "added": r.added.strftime("%b %-d, %Y"), "watched": r.video_id in seen})
        return rows

    all_playlists = []
    for name, pl in playlists.sort_values("added", ascending=False).groupby("playlist", sort=False):
        all_playlists.append({"name": name, "total": len(pl), "videos": playlist_rows(pl)})
    all_playlists.sort(key=lambda p: -p["total"])

    # ---- watch later
    wl = playlists[playlists.playlist == "Watch later"].sort_values("added", ascending=False)
    watch_later = []
    for r in wl.itertuples():
        m = meta.get(r.video_id)
        title = tidy(m["title"]) if m and m.get("title") else None
        ch_name = (m or {}).get("channel") or ""
        row = pd.DataFrame({"title": [title], "channel": [ch_name], "topic": [None]})
        if (title and OFF_LIMITS.search(title + " | " + ch_name)) or excluded(row, ignore_topic=True).iat[0]:
            continue
        watch_later.append({"id": r.video_id, "t": title, "ch": (m or {}).get("channel") or "",
                            "added": r.added.strftime("%b %-d, %Y"), "watched": r.video_id in seen})

    # ---- per-play table for client-side filtering: [card index or -1, day, hour, month, session, dwell s, group]
    order = per_video.sort_values(["n", "last"], ascending=False).reset_index(drop=True)
    cidx = {vid: i for i, vid in enumerate(order.video_id)}
    w["grp"] = w.topic.map(TOPIC_GROUP).fillna(len(GROUPS)).astype(int)
    w["cidx"] = [cidx.get(vid, -1) if ok else -1 for vid, ok in zip(w.video_id, showable)]
    day0 = w.night_of.min()
    month_keys = sorted(w.ts.dt.strftime("%Y-%m").unique())
    plays = [[int(r.cidx), (r.night_of - day0).days, int(r.hour), month_keys.index(r.ts.strftime("%Y-%m")),
              int(r.session), int(r.dwell_s), int(r.grp)] for r in w.itertuples()]

    # ---- daily timeline for the scrub bar (nights roll over at 5 AM)
    daily = w.groupby("night_of").size()
    day_index = pd.date_range(w.night_of.min(), w.night_of.max(), freq="D").date
    daily = daily.reindex(day_index, fill_value=0)

    # ---- channel concentration (Pareto)
    ch_sorted = ch.sort_values("n", ascending=False).n.values
    cum = np.cumsum(ch_sorted) / ch_sorted.sum()
    pareto = {"n": len(ch_sorted), "cum": [round(float(x), 4) for x in cum],
              "for50": int(np.searchsorted(cum, 0.5) + 1), "for80": int(np.searchsorted(cum, 0.8) + 1)}
    ch_stats = (w[w.channel_id.notna()].groupby(["channel_id", "channel"])
                .agg(n=("ts", "size"), skip=("skipped", "mean"), dwell=("dwell_s", "median")).reset_index())
    ch_stats = ch_stats[(ch_stats.n >= 10) & ~ch_stats.channel.str.contains(OFF_LIMITS) & ~ch_stats.channel_id.isin(bad_ch)]
    skip_channels = [{"name": r.channel, "n": int(r.n), "skip": round(float(r.skip), 3), "dwell": int(r.dwell),
                      "topic": ch_topic.get(r.channel_id)} for r in ch_stats.sort_values("skip", ascending=False).itertuples()]

    # ---- rabbit holes: long sessions, what starts them, how topics chain
    long_ids = sessions.sort_values("videos", ascending=False).index
    holes = []
    for sid in long_ids[:12]:
        sw = w[w.session == sid]
        gate = sw[sw.cidx >= 0].head(1)
        holes.append({"session": int(sid), "date": sw.ts.min().strftime("%a, %b %-d"),
                      "start": sw.ts.min().strftime("%-I:%M %p"), "videos": len(sw),
                      "minutes": round(float((sw.ts.max() - sw.ts.min()).total_seconds() / 60)),
                      "gate": int(gate.cidx.iat[0]) if len(gate) else -1,
                      "gate_first": bool(len(gate) and gate.index[0] == sw.index[0]),
                      "seq": sw.grp.tolist()})
    first_of = w.groupby("session").head(1).set_index("session").grp
    is_long = sessions.videos >= 10
    gate_mix = {"long": first_of[is_long[is_long].index].value_counts(normalize=True).reindex(range(5), fill_value=0).round(3).tolist(),
                "short": first_of[is_long[~is_long].index].value_counts(normalize=True).reindex(range(5), fill_value=0).round(3).tolist(),
                "n_long": int(is_long.sum()), "n_short": int((~is_long).sum())}
    nxt_grp = w.grp.shift(-1).where(w.session.shift(-1) == w.session)
    pairs = pd.DataFrame({"a": w.grp, "b": nxt_grp}).dropna()
    trans = pd.crosstab(pairs.a, pairs.b.astype(int)).reindex(index=range(5), columns=range(5), fill_value=0)
    sticky = [round(float(trans.iat[i, i] / trans.iloc[i].sum()), 3) if trans.iloc[i].sum() else 0 for i in range(5)]

    # ---- videos I searched for and clicked (queries stay private; only the clicked video is shown)
    led = []
    for t_search, p in zip(se.ts, pos):
        if p < len(wt) and (wt.iloc[p] - t_search).total_seconds() <= 60:
            led.append(wt.iloc[p])
    led_plays = w[w.ts.isin(set(led)) & (w.cidx >= 0)]
    search_led = led_plays.groupby("cidx").size().sort_values(ascending=False).index.astype(int).tolist()
    search_led_groups = w[w.ts.isin(set(led))].grp.value_counts(normalize=True).reindex(range(5), fill_value=0).round(3).tolist()

    span_days = (w.ts.max() - w.ts.min()).days + 1
    data = {
        "meta": {
            "from": w.ts.min().strftime("%b %-d, %Y"), "to": w.ts.max().strftime("%b %-d, %Y"),
            "days": span_days, "videos": len(w), "unique": int(w.video_id.nunique()),
            "hours": round(w.dwell_s.sum() / 3600), "sessions": len(sessions),
            "median_session_videos": float(sizes.median()),
            "median_session_min": round(float(sessions.minutes.median())),
            "skip_rate": round(float(w.skipped.mean()), 3),
            "repeat_share": round(float((w.nth_view > 1).mean()), 3),
            "rewatched_videos": int((w.groupby("video_id").size() > 1).sum()),
            "channels": int(ch.channel_id.nunique()), "channels_once": int((ch.n == 1).sum()),
            "subs": len(sub_ids), "subs_watched": int(len(sub_watched)),
            "subs_share": round(float(w.channel_id.isin(sub_ids).mean()), 3),
            "searches": len(se), "search_then_watch": round(float((gap_min <= 5).mean()), 3),
            "days_watched": int(w.night_of.nunique()),
            "after_midnight": round(float(w.hour.between(0, 4).mean()), 3),
            "all_plays": n_all, "excluded": n_excluded,
        },
        "shelves": {
            "rewatched": [card(r) for r in rewatched.head(16).itertuples()],
            "late": [card(r) for r in late_top.head(16).itertuples()],
            "by_topic": by_topic,
        },
        "all": [card(r) for r in order.itertuples()],
        "plays": plays,
        "groups": GROUP_NAMES, "topic_group": TOPIC_GROUP,
        "daily": {"from": str(day_index[0]), "n": daily.astype(int).tolist()},
        "pareto": pareto, "skip_channels": skip_channels,
        "holes": holes, "gate_mix": gate_mix, "transitions": trans.astype(int).values.tolist(), "sticky": sticky,
        "search_led": search_led, "search_led_groups": search_led_groups,
        "topics": TOPIC_ORDER,
        "hours": {"days": [d[:3] for d in days], "grid": grid.values.tolist(),
                  "topic_hour": topic_hour.values.tolist(),
                  "band_mix": {b: band_mix[b].round(3).tolist() for b in band_mix.columns},
                  "peak_band": peak_band},
        "months": {"labels": [pd.Timestamp(m + "-01").strftime("%b") for m in months],
                   "partial": [months[0] == w.ts.min().strftime("%Y-%m"), months[-1] == w.ts.max().strftime("%Y-%m")],
                   "share": (tm / tm.sum()).round(3).values.tolist(), "count": tm.astype(int).values.tolist(),
                   "total": month.value_counts().reindex(months).astype(int).tolist()},
        "attention": [{"topic": t, "n": int(r.n), "skip": round(float(r.skip), 3), "dwell": round(float(r.dwell))}
                      for t, r in att.iterrows()],
        "sessions": {
            "bins": [{"label": lab, "n": int(sizes.between(lo, hi).sum()),
                      "videos": int(sizes[sizes.between(lo, hi)].sum())} for lo, hi, lab in size_bins],
            "biggest": {"date": biggest.start.strftime("%A, %b %-d"), "start": biggest.start.strftime("%-I:%M %p"),
                        "end": biggest.end.strftime("%-I:%M %p"), "videos": int(biggest.videos),
                        "minutes": round(float((biggest.end - biggest.start).total_seconds() / 60)),
                        "topics": {k: int(n) for k, n in big_topics.items()}, "strip": big_strip},
        },
        "channels": {
            "loyalty": [{"label": lab, "n": int(ch.n.between(lo, hi).sum()),
                         "videos": int(ch.n[ch.n.between(lo, hi)].sum())} for lo, hi, lab in loyal_bins],
            "top": [{"name": r.channel, "n": int(r.n), "topic": ch_topic.get(r.channel_id),
                     "sub": r.channel_id in sub_ids} for r in top_channels.itertuples()],
        },
        "searches": {"by_hour": se.ts.dt.hour.value_counts().reindex(range(24), fill_value=0).tolist(),
                     "gap_bins": [int((gap_min <= 1).sum()), int(gap_min.between(1, 5, inclusive="right").sum()),
                                  int(gap_min.between(5, 60, inclusive="right").sum()), int((gap_min > 60).sum() + gap_min.isna().sum())]},
        "watch_later": watch_later,
        "playlists": all_playlists,
        "people": people,
        "watch_later_total": len(wl), "watch_later_watched": int(wl.video_id.isin(seen).sum()),
        "pipeline": {
            "cells": sum(log["rows"].values()) and log["rows"]["watches"] + sum(log["watch_dropped"].values()),
            "dropped": log["watch_dropped"], "search_dropped": log["search_dropped"], "rows": log["rows"],
            "deleted": int(v.title.isna().sum()), "off_limits": int(v.off_limits.sum()),
            "agreement": float(re.search(r"held_out_agreement=([\d.]+)", clog).group(1)),
            "sources": {k: int(n) for k, n in v.topic_source.fillna("none").value_counts().items()},
            "session_gap": config.SESSION_GAP_MIN, "dwell_cap": config.DWELL_CAP_MIN,
        },
    }
    config.SITE.mkdir(exist_ok=True)
    (config.SITE / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"site/data.js  {len(json.dumps(data)) / 1024:.0f} KB")
    print(json.dumps(data["meta"], indent=1))
    print("peak by band:", peak_band)


if __name__ == "__main__":
    main()
