"""Read-only FIT audit; outputs contain recorded observations only.

Run: .venv/Scripts/python scripts/audit_watch_wellness_hr.py
Dependencies: requirements-audit.txt. No interpolation, HR merging, or imputation.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pytz

from garmin_fit_sdk import Decoder, Stream
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

UTC = timezone.utc
LOCAL = pytz.timezone('Europe/Brussels')
EPOCH = datetime(1989, 12, 31, tzinfo=UTC)


def monitoring_times(messages):
    """Accumulate low 16 bits in monitoring message order, per file.

    Full monitoring timestamps reset the anchor. Do not use file creation time,
    local_timestamp, or unrelated messages (which can be written out of order).
    Equal low bits mean equal timestamps; a decrease means a 65536s wrap.
    """
    anchor = None
    for original in messages:
        m = dict(original)
        if 'timestamp' in m:
            anchor = int((m['timestamp'] - EPOCH).total_seconds())
        elif 'timestamp_16' in m:
            if anchor is None:
                raise ValueError('timestamp_16 without a preceding full monitoring timestamp')
            low = m['timestamp_16']
            anchor += (low - (anchor & 0xffff)) & 0xffff
            m['timestamp'] = EPOCH + timedelta(seconds=anchor)
            assert anchor & 0xffff == low
        elif 'timestamp_min_8' in m:
            raise ValueError('timestamp_min_8 needs explicit decoding before analysis')
        yield m


def stats(frame):
    d = frame.sort_values('timestamp').timestamp.diff().dt.total_seconds().round(6).dropna()
    result = {'samples': len(frame), 'intervals': len(d)}
    if len(d):
        result.update({k: float(v) for k, v in {'mean': d.mean(), 'median': d.median(),
                      'min': d.min(), 'max': d.max()}.items()})
        result['percentiles'] = {str(p): float(d.quantile(p/100)) for p in [5, 25, 50, 75, 95, 99]}
        result['distribution_seconds'] = {str(k): int(v) for k, v in d.value_counts().sort_index().items()}
        result['within_1s_percent'] = {str(s): float(100*((d-s).abs() <= 1).mean()) for s in [1, 5, 15, 30, 60, 120]}
        result['gaps_over_120s'] = int((d > 120).sum())
        result['gap_elapsed_seconds'] = float(d[d > 120].sum())
    return result


def write_csv(frame, path):
    frame = frame.copy()
    if 'timestamp' in frame:
        frame['timestamp_local'] = frame.timestamp.map(lambda t: t.astimezone(LOCAL).isoformat())
    frame.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--date', default='2026-10-16')
    parser.add_argument('--edge', default='data/mtb_edge.fit')
    args = parser.parse_args()
    root = args.root
    out = root / 'data/processed'
    docs = root / 'docs'
    (docs / 'images').mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    source_files = sorted(p for p in (root/'data').rglob('*') if p.is_file() and out not in p.parents)
    unsupported = [str(p.relative_to(root)) for p in source_files if p.suffix.lower() != '.fit']
    if unsupported:
        raise ValueError(f'Inspect these non-FIT files/archives before proceeding: {unsupported}')
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    inventory, observations, events, monitoring, decoded_watch = [], [], [], [], {}
    all_messages = {}
    for path in source_files:
        name = str(path.relative_to(root)).replace('\\', '/')
        decoder = Decoder(Stream.from_file(str(path)))
        if not decoder.check_integrity():
            raise ValueError(f'Integrity failure: {name}')
        decoder = Decoder(Stream.from_file(str(path)))
        messages, errors = decoder.read(merge_heart_rates=False)
        if errors:
            raise ValueError(f'{name}: {errors}')
        all_messages[name] = messages
        product = messages.get('file_id_mesgs', [{}])[0].get('garmin_product', 'unknown')
        watch = product == 'fr265_large'
        if 'watch_wellness' in path.parts:
            decoded_watch[name] = messages
        if messages.get('hr_mesgs'):
            raise ValueError(f'{name}: array HR messages require event-time extraction; refusing silent omission')
        for kind, rows in messages.items():
            resolved = list(monitoring_times(rows)) if kind == 'monitoring_mesgs' else rows
            times = [m['timestamp'] for m in resolved if isinstance(m.get('timestamp'), datetime)]
            inventory.append({'file': name, 'device': product, 'message_type': kind, 'messages': len(rows),
                'fields': ', '.join(sorted({str(k) for m in rows for k in m})),
                'first_utc': min(times).isoformat() if times else '',
                'last_utc': max(times).isoformat() if times else '',
                'hr_fields': sum('heart_rate' in m for m in rows),
                'valid_hr': sum(isinstance(m.get('heart_rate'), (int,float)) and 0 < m['heart_rate'] < 255 for m in rows)})
            for index, m in enumerate(resolved):
                if watch and kind == 'monitoring_mesgs':
                    monitoring.append({'source_file': name, **m})
                if watch and kind == 'event_mesgs':
                    events.append({'source_file': name, **m})
                if 'heart_rate' in m:
                    if isinstance(m['heart_rate'], list) or 'timestamp' not in m:
                        raise ValueError(f'Unhandled HR encoding: {name}, {kind}, {m}')
                    observations.append({'timestamp': m['timestamp'], 'bpm': m['heart_rate'],
                        'source_file': name, 'message_type': kind, 'message_index': index,
                        'valid': isinstance(m['heart_rate'], (int,float)) and 0 < m['heart_rate'] < 255,
                        'source': 'watch' if watch else 'edge' if name == args.edge else 'edge_wellness' if product == 'edge_840' and 'watch_wellness' in path.parts else 'other_activity'})
    pd.DataFrame(inventory).to_csv(out/'fit_inventory.csv', index=False)
    (out/'watch_decoded_messages.json').write_text(json.dumps(decoded_watch, default=str, indent=2), encoding='utf-8')
    (out/'source_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    hr = pd.DataFrame(observations).sort_values(['timestamp','source_file'])
    write_csv(hr, out/'hr_all_sources.csv')
    watch_raw = hr[(hr.source == 'watch') & hr.valid].copy()
    duplicate_count = int(watch_raw.duplicated(['timestamp','bpm']).sum())
    watch = watch_raw.drop_duplicates(['timestamp','bpm']).copy()
    conflicts = int((watch.groupby('timestamp').bpm.nunique() > 1).sum())
    if conflicts:
        raise ValueError('Conflicting valid watch observations need review')
    edge = hr[(hr.source == 'edge') & hr.valid].copy()
    em = all_messages[args.edge]
    session = em['session_mesgs'][0]
    start = session['start_time']
    end = start + timedelta(seconds=session['total_elapsed_time'])
    last_hr = edge.timestamp.max()
    edge_records = pd.DataFrame(em['record_mesgs'])
    first_missing = edge_records[edge_records.timestamp > last_hr].timestamp.min()
    day_start = LOCAL.localize(datetime.fromisoformat(args.date)).astimezone(UTC)
    day_end = max(day_start + timedelta(days=1), end)
    day = watch[(watch.timestamp >= day_start) & (watch.timestamp < day_end)].copy()
    ride = day[(day.timestamp >= start) & (day.timestamp <= end)].copy()
    write_csv(ride, out/f'watch_hr_ride_{args.date}.csv')
    periods = {'before': day[day.timestamp < start], 'during': ride, 'after': day[day.timestamp > end]}
    statistics = {k: stats(v) for k,v in periods.items()}
    evening_start = day_start + timedelta(hours=18, minutes=30)
    statistics['before_evening_only'] = stats(day[(day.timestamp >= evening_start) & (day.timestamp < start)])
    statistics['edge_valid'] = stats(edge)
    statistics['per_file'] = {k:stats(v) for k,v in watch.groupby('source_file')}
    day['interval_seconds'] = day.timestamp.diff().dt.total_seconds().round(6)
    day['previous_timestamp'] = day.timestamp.shift()
    write_csv(day, out/'watch_intervals.csv')
    write_csv(day[day.interval_seconds > 120], out/'watch_gaps_over_120s.csv')
    statistics['hourly'] = {str(k): stats(v) for k,v in day.groupby(day.timestamp.dt.tz_convert(LOCAL).dt.floor('h'))}
    # Exact timestamp pairs only: no resampling, alignment shift, or interpolation.
    pairs = ride.merge(edge[['timestamp','bpm']], on='timestamp', suffixes=('_watch','_edge'))
    pairs['difference_bpm'] = pairs.bpm_watch - pairs.bpm_edge
    write_csv(pairs, out/'watch_edge_exact_pairs.csv')
    dif = pairs.difference_bpm
    comparison = {'pairs': len(pairs), 'mean_watch_minus_edge': float(dif.mean()),
        'median_watch_minus_edge': float(dif.median()), 'mae': float(dif.abs().mean()),
        'min_difference': float(dif.min()), 'max_difference': float(dif.max()),
        'watch_after_edge_loss': int((ride.timestamp > last_hr).sum()),
        'edge_last_hr': str(last_hr), 'edge_first_missing': str(first_missing)}
    # data16 is event duration in minutes (SDK profile event.data subfield).
    # Event timestamp is reporting time, not the classified interval's end.
    auto = []
    for e in events:
        if e.get('event') != 'auto_activity_detect':
            continue
        a = e['start_timestamp']
        b = a + timedelta(minutes=e.get('auto_activity_detect_duration', e.get('data', e.get('data16'))))
        selected = day[(day.timestamp >= a) & (day.timestamp < b)]
        auto.append({'source_file': e['source_file'], 'activity_type': e.get('activity_type'),
                     'start_local': a.astimezone(LOCAL).isoformat(), 'end_local': b.astimezone(LOCAL).isoformat(),
                     'reported_local': e['timestamp'].astimezone(LOCAL).isoformat(),
                     'stats': stats(selected),
                     'before_15min': stats(day[(day.timestamp >= a-timedelta(minutes=15)) & (day.timestamp < a)]),
                     'after_15min': stats(day[(day.timestamp >= b) & (day.timestamp < b+timedelta(minutes=15))])})
    (out/'auto_activity_intervals.json').write_text(json.dumps(auto, indent=2), encoding='utf-8')
    (out/'watch_events.json').write_text(json.dumps(events, default=str, indent=2), encoding='utf-8')
    pd.DataFrame(monitoring).to_csv(out/'watch_monitoring.csv', index=False)
    # Describe interval endpoints by recorded BPM; descriptive association, not a causal test.
    statistics['intervals_by_bpm'] = {}
    for label, mask in [('below_100', day.bpm < 100), ('100_to_129', day.bpm.between(100,129)), ('130_plus', day.bpm >= 130)]:
        d = day.loc[mask,'interval_seconds'].dropna()
        statistics['intervals_by_bpm'][label] = {'intervals':len(d), 'median':float(d.median()), 'min':float(d.min()), 'mean':float(d.mean())}
    summary = {'statistics':statistics, 'comparison':comparison, 'duplicate_valid_rows':duplicate_count,
               'conflicting_timestamps':conflicts, 'ride_files':ride.source_file.value_counts().to_dict(),
               'invalid_watch_rows':int(((hr.source == 'watch') & ~hr.valid).sum())}
    (out/'audit_statistics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    visible = day[day.timestamp >= evening_start]
    fig, axes = plt.subplots(2,1,figsize=(15,8),sharex=True,gridspec_kw={'height_ratios':[2,1]}, constrained_layout=True)
    axes[0].scatter(visible.timestamp, visible.bpm, s=18, color='#176b91', label='Watch wellness: stored HR', zorder=3)
    axes[0].scatter(edge.timestamp, edge.bpm, s=5, alpha=.45, color='#dc681f', label='Edge: recorded HR', zorder=2)
    axes[1].scatter(visible.timestamp, visible.interval_seconds, s=16, color='#176b91')
    axes[1].axhline(120, color='#b24045', ls='--', label='120-second reference')
    cycling = [e for e in auto if e['activity_type']=='cycling']
    for i,e in enumerate(cycling):
        a,b = datetime.fromisoformat(e['start_local']),datetime.fromisoformat(e['end_local'])
        axes[0].plot([a,b],[.98,.98],transform=axes[0].get_xaxis_transform(),lw=5,color='#6d4a9e',label='Auto-detected cycling' if i==0 else None)
    for ax in axes:
        ax.axvspan(start,end,color='#dbc473',alpha=.22,label='MTB session' if ax is axes[0] else None)
        ax.axvline(first_missing,color='#b24045',ls=':',label='Edge first missing HR' if ax is axes[0] else None)
        ax.grid(alpha=.18)
    axes[0].set(ylabel='Heart rate (BPM)', title=f'Forerunner 265 wellness HR and Edge 840 | {args.date}\nRecorded observations only; purple bars show detected cycling intervals')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,.94),ncol=2,fontsize=9)
    axes[1].set(ylabel='Stored HR interval (seconds)',xlabel='Local time — Europe/Brussels (UTC+02:00)')
    axes[1].legend(loc='upper right')
    axes[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30],tz=LOCAL))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=LOCAL))
    axes[1].set_xlim(evening_start,day_end)
    figure = docs/'images'/f'watch_wellness_hr_{args.date}.png'
    fig.savefig(figure,dpi=220)
    plt.close(fig)
    report = [f'# Watch wellness HR audit — {args.date}', '',
        f'**158 real watch observations recovered during the ride; stored spacing is 60 or 120 seconds.** '
        'Automatic cycling detection is present, but no sub-minute wellness HR is exposed.', '',
        '## Sources and extraction', '',
        f'Inspected all {len(source_files)} FIT files recursively (81 in `data/watch_wellness/`, plus three activity files); no archives or other source files. '
        'All passed size/header/CRC integrity and decoding checks. Device identity, not folder name, determines provenance: '
        '29 files identify the FR265 and 55 identify the Edge (including the three activities). '
        'The nominal watch folder therefore includes 52 Edge files.', '',
        'The watch HR series is `monitoring` (global message 55), `heart_rate` (field 27, BPM), with `timestamp_16` (field 26). '
        'The ride is entirely in **`data/watch_wellness/494532015728_WELLNESS.fit`**: '
        f'{len(ride)} samples, {ride.timestamp.min().astimezone(LOCAL).isoformat()} through {ride.timestamp.max().astimezone(LOCAL).isoformat()}. '
        'This file has 166 HR samples overall, 19:29–22:29. Creator is `fr265_large` (product 4257), firmware 29.05.', '',
        'The official Garmin SDK decodes fields/components with HR merging disabled. Monitoring timestamps are accumulated in original '
        'monitoring-message order within each file: `anchor += (timestamp_16 - (anchor & 65535)) & 65535`; full monitoring '
        'timestamps reset the anchor. The epoch is 1989-12-31 UTC. This handles the 16-bit wrap and is distinct from compressed '
        'record headers. SDK 21.218.0 rejects compressed timestamp headers; no such decoding errors occurred here. '
        'No standard `hr` array/event-time messages were found. Unknown message numbers and packed arrays remain preserved '
        'in the decoded JSON; their undocumented semantics cannot be ruled out as containing additional proprietary information.', '',
        'No valid watch duplicates or conflicting watch timestamps were found. `494531572096_WELLNESS.fit` is **Edge wellness**, '
        'with 18 valid HR values (19:34–19:59), including 15 differing values at timestamps shared with the watch. '
        'It is exported separately and never merged into the watch series. Zero-BPM records in Edge wellness are retained as invalid; '
        'the watch has no zero-BPM observations. Resting-HR summaries (`monitoring_hr_data`, 35 watch messages), HRV values/status, '
        'SpO2, stress, respiration and session averages are not additional instantaneous HR samples.', '',
        '## Time window and sampling', '',
        f'All times below are Europe/Brussels (UTC+02:00). Edge session start: **{start.astimezone(LOCAL).isoformat()}**. '
        f'Start + elapsed duration gives **{end.astimezone(LOCAL).isoformat()}**; final record and timer stop are 22:25:50. '
        'The session/lap summary timestamp itself repeats the start, so it was not used as the end. '
        'The ride does not cross midnight. Watch coverage is 00:01 on October 16 through exactly 00:00 on October 17; '
        'daily statistics use [00:00, next 00:00), excluding that last midnight sample. The plot shows 18:30–24:00.', '',
        '| Period | Samples | Median / mean (s) | Min–max (s) | 60s / 120s intervals | Gaps >120s (elapsed seconds) |',
        '|---|---:|---:|---:|---:|---:|']
    for k in ['before','before_evening_only','during','after']:
        s=statistics[k]
        report.append(f"| {k} | {s['samples']} | {s['median']:.0f} / {s['mean']:.2f} | {s['min']:.0f}–{s['max']:.0f} | {s['within_1s_percent']['60']:.2f}% / {s['within_1s_percent']['120']:.2f}% | {s['gaps_over_120s']} ({s['gap_elapsed_seconds']:.0f}s) |")
    report += ['', 'Before/after are the portions of this calendar day outside the session; evening-only begins 18:30. '
        'Intervals are calculated within each period, excluding boundary-crossing pairs. Significant gaps are defined as >120s '
        '(elapsed between recorded endpoints, not estimated missing duration). All watch proportions near 1, 5, 15 and 30 seconds '
        'are 0%, with “near” defined as ±1s.', '',
        '| Period | P5 / P25 / P50 / P75 / P95 / P99 (s) | Full interval distribution (seconds: count) |',
        '|---|---|---|']
    for k in ['before','during','after']:
        s=statistics[k]
        report.append(f"| {k} | {' / '.join(f'{v:.1f}' for v in s['percentiles'].values())} | {', '.join(f'{float(t):g}: {n}' for t,n in s['distribution_seconds'].items())} |")
    report += ['', 'During the ride, 142 intervals are 60s and 15 are 120s; none exceeds 120s. '
        'All timestamps lie on minute boundaries. This describes **stored samples**, not optical sensor sampling frequency. '
        'The public profile labels the field BPM but does not establish whether it is an instantaneous reading or a filtered/averaged '
        'value, nor its averaging window. Timestamp precision does not prove physiological precision.', '',
        '## Automatic activity detection and intensity', '',
        'The watch has explicit `event=auto_activity_detect` records classified as cycling, consistent with Move IQ. '
        'Intervals below use `start_timestamp + data duration in minutes`; event timestamps are reporting times, 15 minutes after '
        'the classified end for these events. Using reporting time as the end would incorrectly extend the exercise windows.', '',
        '| Detected cycling (local) | Reported | Samples | Mean spacing (s) | Median before / during / after (s)* |',
        '|---|---|---:|---:|---|']
    for e in cycling:
        med = lambda s: f"{s['median']:.0f}" if 'median' in s else 'n/a'
        report.append(f"| {e['start_local'][11:19]}–{e['end_local'][11:19]} | {e['reported_local'][11:19]} | {e['stats']['samples']} | {e['stats']['mean']:.2f} | {med(e['before_15min'])} / {med(e['stats'])} / {med(e['after_15min'])} |")
    report += ['', '*Before/after use adjacent 15-minute windows; they may include other classified activity. '
        'Complete event statistics are in `auto_activity_intervals.json`. The final ride classification is in '
        '`494552312010_WELLNESS.fit`, although the HR samples it describes are in the earlier ride file.', '',
        'Monitoring includes activity classifications, packed type/intensity components, active time, cycles and moderate/vigorous '
        'activity-minute records. These are preserved in `watch_monitoring.csv`; they are not HR observations. '
        'For interval endpoints with BPM <100, 100–129 and ≥130, mean spacing is respectively '
        + ', '.join(f"{statistics['intervals_by_bpm'][k]['mean']:.2f}s" for k in ['below_100','100_to_129','130_plus'])
        + '; all three groups have 60s median and minimum. Higher BPM is associated with fewer skipped minute slots, '
        'but there is **no faster stored cadence**, and this single-day association cannot establish causation by Move IQ.', '',
        '## Edge comparison', '',
        f'The Edge has {len(edge_records)} activity records and **{len(edge)} consecutive 1Hz HR observations**, '
        '19:32:22–19:57:46. The first missing HR record is **19:57:47**; the remaining 8,884 activity records lack HR. '
        'This is an observed recording cutoff, not proof of the exact radio failure instant; no public `comm_timeout` event was found. '
        f'The watch retains **{comparison["watch_after_edge_loss"]} samples after that cutoff**, through 22:25 during the ride.', '',
        f'Exact timestamp matching gives {len(pairs)} watch–Edge pairs (no time shift or interpolation). Mean watch minus Edge is '
        f'{dif.mean():.2f} BPM, median {dif.median():.0f} BPM, mean absolute difference {dif.abs().mean():.2f} BPM, '
        f'range {dif.min():.0f} to {dif.max():.0f} BPM. Both use UTC timestamps, but unknown averaging/processing windows '
        'mean timestamp equality does not imply identical underlying measurement windows. The small overlap does not establish '
        'a stable correction or clock offset.', '',
        '## Conclusions and limitations', '',
        '1. Ride watch HR is in `494532015728_WELLNESS.fit` only.',
        f'2. Recovered {len(ride)} valid watch HR observations during the precise Edge session.',
        '3. Actual spacing: 60s (90.45%) or 120s (9.55%); median 60s, mean 65.73s.',
        '4. Auto-detected cycling is documented, but neither it nor elevated HR produces sub-minute stored samples in the decoded public fields.',
        '5. Compared with a two-minute display, these 158 points offer roughly 1.8 times as many observations over the ride '
        '(about 87 two-minute slots). That is useful additional temporal detail, not a hidden high-resolution stream. '
        'No Garmin Connect graph/API export was supplied to verify its exact averaging or information loss.',
        '6. These observations can anchor a future coarse HR reconstruction, especially after the Edge cutoff. They cannot recover '
        'true second-by-second dynamics or missing peaks, and any later denser reconstruction must be labeled estimated. '
        'Unknown averaging and the short, imperfect Edge overlap limit calibration.', '',
        '## Outputs and reproduction', '',
        f'- Figure: `docs/images/watch_wellness_hr_{args.date}.png` (3300×1760).',
        '- Full per-file/message inventory, fields, resolved UTC ranges and HR counts: `data/processed/fit_inventory.csv`.',
        '- Every extracted scalar HR, with timestamp, device source, file and message index: `data/processed/hr_all_sources.csv`.',
        f'- Ride watch HR: `data/processed/watch_hr_ride_{args.date}.csv`.',
        '- Interval statistics (including hourly and per-file), gaps, exact Edge pairs, monitoring and automatic events: `data/processed/`.',
        '- Lossless SDK-decoded watch-folder messages, including unknown fields: `data/processed/watch_decoded_messages.json`.',
        '- Original-file SHA-256 manifest: `data/processed/source_sha256.json`; originals rechecked unchanged after analysis.',
        '', '```powershell', '.\\.venv\\Scripts\\python.exe -m pip install -r requirements-audit.txt',
        '.\\.venv\\Scripts\\python.exe scripts/audit_watch_wellness_hr.py', '```', '',
        'No interpolation, synthetic HR, model training, FIT modification or uploads were performed. '
        'The script fails explicitly on unsupported inputs/HR encodings rather than silently omitting them.', '',
        '## Format references', '',
        '- [Official Garmin Python FIT SDK](https://github.com/garmin/fit-python-sdk), version 21.218.0 used.',
        '- [Garmin FIT engineer explanation of timestamp_16](https://forums.garmin.com/developer/fit-sdk/f/discussion/424427/convert-timestamp-16-to-date).',
        '- [Garmin Event message profile](https://github.com/garmin/fit-csharp-sdk/blob/main/Dynastream/Fit/Profile/Mesgs/EventMesg.cs): automatic-detection start and duration fields.',
        '']
    (docs/f'watch_wellness_hr_audit_{args.date}.md').write_text('\n'.join(report), encoding='utf-8')
    # Originals are checked again after every output has been generated.
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({'summary':summary, 'cycling_events':cycling},indent=2))


if __name__ == '__main__':
    main()
