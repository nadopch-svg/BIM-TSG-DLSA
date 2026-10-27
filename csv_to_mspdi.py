#!/usr/bin/env python3
"""
DLSA_Schedule.csv  ->  MS Project XML (MSPDI) + mapping CSV

BEXEL Manager는 Schedule Editor > Import > "From MS Project" / "From Primavera P6"
경로로 일정을 가져온다. CSV는 직접 import 대상이 아니므로 이 스크립트가 브릿지 역할을 한다.

출력
  1) <out>.xml          : BEXEL이 읽을 MSPDI 일정 (레이어 = 태스크, 레이어 간 FS 링크)
  2) <out>_mapping.csv  : GlobalId -> DLSA_Key (IFC 속성 주입 + 링크 검증에 사용)

사용 예
  python csv_to_mspdi.py DLSA_Schedule.csv --start 2026-11-02 --out out/DLSA_Dense
  python csv_to_mspdi.py DLSA_Schedule.csv --mode component --days-per-layer 2
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

NS = "http://schemas.microsoft.com/project"
ET.register_namespace("", NS)

# MS Project 고정 FieldID (Text1/Text2/Text3)
FIELD_TEXT1, FIELD_TEXT2, FIELD_TEXT3 = 188743731, 188743734, 188743737

# 웹 앱 버전에 따라 컬럼명이 다를 수 있으므로 별칭으로 탐색
ALIASES = {
    "guid": ["GlobalId", "GUID", "Guid", "guid", "global_id", "IfcGUID"],
    "layer": ["Layer", "layer", "DLSA_Layer", "layer_index", "LayerIndex", "layer_id"],
    "name": ["Name", "name", "ComponentName", "element_name"],
    "type": ["IfcType", "Type", "type", "Category", "ifc_class"],
    "duration": ["Duration", "duration", "duration_days", "Days"],
    "floor": ["Floor", "Storey", "Level", "floor", "storey"],
}

HOURS_PER_DAY = 8
DAY_START, DAY_END = dt.time(8, 0), dt.time(17, 0)


def resolve_columns(df: pd.DataFrame, overrides: dict[str, str | None]) -> dict[str, str | None]:
    cols = {}
    for key, cands in ALIASES.items():
        if overrides.get(key):
            if overrides[key] not in df.columns:
                sys.exit(f"[ERR] --{key}-col '{overrides[key]}' 컬럼이 CSV에 없습니다: {list(df.columns)}")
            cols[key] = overrides[key]
            continue
        cols[key] = next((c for c in cands if c in df.columns), None)
    for required in ("guid", "layer"):
        if cols[required] is None:
            sys.exit(f"[ERR] 필수 컬럼({required})을 찾지 못했습니다. --{required}-col 로 지정하세요. "
                     f"현재 컬럼: {list(df.columns)}")
    return cols


def add_workdays(start: dt.date, n: int) -> dt.date:
    """start(근무일)부터 n 근무일째 되는 날짜 (n=1이면 start 자신). 토/일 제외."""
    d, counted = start, 0
    while True:
        if d.weekday() < 5:
            counted += 1
            if counted == n:
                return d
        d += dt.timedelta(days=1)


def next_workday(d: dt.date) -> dt.date:
    d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def first_workday(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def iso(d: dt.date, t: dt.time) -> str:
    return dt.datetime.combine(d, t).strftime("%Y-%m-%dT%H:%M:%S")


def sub(parent, tag, text=None):
    el = ET.SubElement(parent, f"{{{NS}}}{tag}")
    if text is not None:
        el.text = str(text)
    return el


def build_calendar(project):
    cals = sub(project, "Calendars")
    cal = sub(cals, "Calendar")
    sub(cal, "UID", 1); sub(cal, "Name", "Standard"); sub(cal, "IsBaseCalendar", 1)
    wds = sub(cal, "WeekDays")
    for day_type in range(1, 8):  # 1=일 ... 7=토
        wd = sub(wds, "WeekDay")
        sub(wd, "DayType", day_type)
        working = day_type not in (1, 7)
        sub(wd, "DayWorking", int(working))
        if working:
            wts = sub(wd, "WorkingTimes")
            for a, b in (("08:00:00", "12:00:00"), ("13:00:00", "17:00:00")):
                wt = sub(wts, "WorkingTime"); sub(wt, "FromTime", a); sub(wt, "ToTime", b)


def build_extended_attr_defs(project):
    eas = sub(project, "ExtendedAttributes")
    for fid, fname, alias in ((FIELD_TEXT1, "Text1", "DLSA_Key"),
                              (FIELD_TEXT2, "Text2", "DLSA_Count"),
                              (FIELD_TEXT3, "Text3", "DLSA_Source")):
        ea = sub(eas, "ExtendedAttribute")
        sub(ea, "FieldID", fid); sub(ea, "FieldName", fname); sub(ea, "Alias", alias)


def add_task(tasks, uid, name, level, outline, start, finish, days, *,
             summary=False, preds=(), text1=None, text2=None, text3=None, pin_start=False):
    t = sub(tasks, "Task")
    sub(t, "UID", uid); sub(t, "ID", uid); sub(t, "Name", name)
    sub(t, "Type", 1)                       # Fixed Duration
    sub(t, "IsNull", 0)
    sub(t, "OutlineNumber", outline); sub(t, "OutlineLevel", level)
    sub(t, "Start", iso(start, DAY_START)); sub(t, "Finish", iso(finish, DAY_END))
    sub(t, "Duration", f"PT{days * HOURS_PER_DAY}H0M0S"); sub(t, "DurationFormat", 7)
    sub(t, "Summary", int(summary)); sub(t, "Milestone", 0); sub(t, "Manual", 0)
    if pin_start:                           # Start No Earlier Than
        sub(t, "ConstraintType", 4); sub(t, "ConstraintDate", iso(start, DAY_START))
    else:                                   # As Soon As Possible
        sub(t, "ConstraintType", 0)
    sub(t, "CalendarUID", -1)
    for p in preds:
        pl = sub(t, "PredecessorLink")
        sub(pl, "PredecessorUID", p); sub(pl, "Type", 1)  # 1 = Finish-to-Start
        sub(pl, "CrossProject", 0); sub(pl, "LinkLag", 0); sub(pl, "LagFormat", 7)
    for fid, val in ((FIELD_TEXT1, text1), (FIELD_TEXT2, text2), (FIELD_TEXT3, text3)):
        if val is not None:
            ea = sub(t, "ExtendedAttribute"); sub(ea, "FieldID", fid); sub(ea, "Value", val)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default="DLSA_Schedule_MSP", help="출력 경로 접두어 (확장자 제외)")
    ap.add_argument("--project-name", default="DLSA Schedule")
    ap.add_argument("--start", default=None, help="착공일 YYYY-MM-DD (기본: 다음 월요일)")
    ap.add_argument("--mode", choices=["layer", "component"], default="layer",
                    help="layer: 레이어당 1태스크(권장) / component: 부재당 1태스크(소형 모델용)")
    ap.add_argument("--days-per-layer", type=int, default=1,
                    help="Duration 컬럼이 없을 때 레이어 1개의 공기(근무일)")
    ap.add_argument("--key-prefix", default="DLSA_L")
    for k in ALIASES:
        ap.add_argument(f"--{k}-col", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols = resolve_columns(df, {k: getattr(args, f"{k}_col") for k in ALIASES})
    df = df.dropna(subset=[cols["guid"], cols["layer"]]).copy()
    df[cols["guid"]] = df[cols["guid"]].astype(str).str.strip()

    dupes = df[cols["guid"]].duplicated().sum()
    if dupes:
        print(f"[WARN] GlobalId 중복 {dupes}건 -> 첫 행만 사용", file=sys.stderr)
        df = df.drop_duplicates(subset=[cols["guid"]])

    # 레이어 값(0-based/1-based/문자열 모두 허용)을 정렬된 순번으로 정규화
    raw_layers = sorted(df[cols["layer"]].unique(), key=lambda v: (float(v) if str(v).replace('.', '', 1).isdigit() else math.inf, str(v)))
    width = max(3, len(str(len(raw_layers))))
    layer_key = {v: f"{args.key_prefix}{i + 1:0{width}d}" for i, v in enumerate(raw_layers)}
    df["DLSA_Key"] = df[cols["layer"]].map(layer_key)

    start = first_workday(dt.date.fromisoformat(args.start)) if args.start else \
        first_workday(dt.date.today() + dt.timedelta(days=(7 - dt.date.today().weekday()) % 7 or 7))

    project = ET.Element(f"{{{NS}}}Project")
    sub(project, "SaveVersion", 14)
    sub(project, "Name", args.project_name); sub(project, "Title", args.project_name)
    sub(project, "ScheduleFromStart", 1)
    sub(project, "StartDate", iso(start, DAY_START))
    sub(project, "CalendarUID", 1)
    sub(project, "DefaultStartTime", "08:00:00"); sub(project, "DefaultFinishTime", "17:00:00")
    sub(project, "MinutesPerDay", HOURS_PER_DAY * 60); sub(project, "MinutesPerWeek", HOURS_PER_DAY * 60 * 5)
    sub(project, "DaysPerMonth", 20); sub(project, "DurationFormat", 7)
    build_extended_attr_defs(project)
    build_calendar(project)
    tasks = sub(project, "Tasks")

    # 레이어별 공기 결정: 같은 레이어 부재는 병렬이므로 max(duration)
    layer_days = {}
    for raw, key in layer_key.items():
        grp = df[df[cols["layer"]] == raw]
        if cols["duration"]:
            d = pd.to_numeric(grp[cols["duration"]], errors="coerce").max()
            layer_days[key] = max(1, int(math.ceil(d))) if pd.notna(d) else args.days_per_layer
        else:
            layer_days[key] = args.days_per_layer

    # 날짜 계산 (레이어는 위상정렬 순서이므로 직렬 FS)
    windows, cursor = {}, start
    for key in layer_key.values():
        fin = add_workdays(cursor, layer_days[key])
        windows[key] = (cursor, fin)
        cursor = next_workday(fin)
    project_finish = windows[list(layer_key.values())[-1]][1]
    sub(project, "FinishDate", iso(project_finish, DAY_END))
    # FinishDate는 Tasks 앞에 와야 스키마 순서가 맞으므로 위치 조정
    fd = project.find(f"{{{NS}}}FinishDate"); project.remove(fd); project.insert(5, fd)

    total_days = sum(layer_days.values())
    add_task(tasks, 0, args.project_name, 0, "0", start, project_finish, total_days, summary=True)

    uid = 1
    if args.mode == "layer":
        prev_uid = None
        for i, (raw, key) in enumerate(layer_key.items(), start=1):
            s, f = windows[key]
            n = int((df["DLSA_Key"] == key).sum())
            add_task(tasks, uid, key, 1, str(i), s, f, layer_days[key],
                     preds=[prev_uid] if prev_uid else [],
                     text1=key, text2=str(n), text3=f"layer={raw}")
            prev_uid, uid = uid, uid + 1
    else:
        if len(df) > 3000:
            print(f"[WARN] component 모드에서 태스크 {len(df)}개 생성 -> BEXEL 링크 단계가 매우 느려질 수 있습니다.",
                  file=sys.stderr)
        for i, (raw, key) in enumerate(layer_key.items(), start=1):
            s, f = windows[key]
            add_task(tasks, uid, key, 1, str(i), s, f, layer_days[key], summary=True, text1=key)
            parent_uid, uid = uid, uid + 1
            grp = df[df["DLSA_Key"] == key]
            for j, row in enumerate(grp.itertuples(index=False), start=1):
                r = row._asdict()
                label = r.get(cols["name"]) if cols["name"] else None
                label = f"{label} [{r[cols['guid']]}]" if label else r[cols["guid"]]
                add_task(tasks, uid, label, 2, f"{i}.{j}", s, f, layer_days[key],
                         text1=r[cols["guid"]], text2=key, text3=str(r.get(cols["type"], "")) if cols["type"] else None,
                         pin_start=True)
                uid += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(project, space="  ")
    ET.ElementTree(project).write(out.with_suffix(".xml"), encoding="utf-8", xml_declaration=True)

    map_cols = [cols["guid"], "DLSA_Key", cols["layer"]] + [c for c in (cols["type"], cols["floor"]) if c]
    df[map_cols].rename(columns={cols["guid"]: "GlobalId"}).to_csv(
        out.with_name(out.name + "_mapping.csv"), index=False, encoding="utf-8-sig")

    print(f"[OK] {out.with_suffix('.xml')}  (mode={args.mode}, layers={len(layer_key)}, "
          f"components={len(df)}, {start} ~ {project_finish})")
    print(f"[OK] {out.with_name(out.name + '_mapping.csv')}")


if __name__ == "__main__":
    main()
