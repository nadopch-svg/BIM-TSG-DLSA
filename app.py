# -*- coding: utf-8 -*-
"""
BIM 4D 공정 자동화 웹앱 — TSG(Topological Semantic Graph) × DLSA(Dynamic Layering Scheduling Algorithm)

참고 논문: Zhang et al., "BIM-based construction scheduling optimization through graph neural
network-driven spatial semantic reasoning", J. Ind. Inf. Integr. 51 (2026) 101088.

────────────────────────────────────────────────────────────────────────────
설치
    pip install "streamlit>=1.37" pandas numpy networkx altair matplotlib ifcopenshell
    # pip 설치가 안 되는 환경이면: conda install -c conda-forge ifcopenshell
실행
    streamlit run app.py
────────────────────────────────────────────────────────────────────────────
구성
    1) IFC 파싱  : 부재 추출 → 공종 분류 → 형상(월드 좌표) AABB → 층 매핑 → IfcRelFillsElement
    2) TSG 구축  : sweep-and-prune 근접쌍 탐색 → 포함/매립/접촉 판정 → 속성 기반 방향 보정
                   (상하 지지·공종 위계·IFC 개구부 채움·중력 지지 추정) → 순환 제거(최약 엣지)
    3) DLSA      : 진입차수 0 후보 → 우선순위(지지값/영향값/임계경로) → N_max 절단
                   → 동일 레이어 공간 충돌 회피 → 작업일 캘린더(주 5/6/7일 + 휴일) 변환
    4) 돌발 변수 : 채팅 자연어 → 이벤트(착수 지연·공기 연장·장비 고장·작업 중지·동시작업 한도)
                   → 기준일 이전 착수 레이어 동결 → 잔여 서브그래프만 DLSA 재계층화
    * IFC 파일이 없어도 사이드바의 '샘플 모델'로 전체 흐름을 체험할 수 있습니다.
"""
from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import random
import re
import tempfile
import time
import unicodedata
from datetime import date, datetime

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
    ALT_OK = True
except Exception:  # altair는 streamlit 의존성이라 보통 설치되어 있음
    ALT_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")  # 서버 렌더링(GUI 없음)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (projection="3d" 등록)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    MPL_OK, MPL_ERR = True, ""
except Exception as _e:
    MPL_OK, MPL_ERR = False, str(_e)

try:
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as ifc_el
    import ifcopenshell.util.placement as ifc_pl
    import ifcopenshell.util.unit as ifc_unit
    IFC_OK, IFC_ERR = True, ""
except Exception as _e:  # 샘플 모델은 ifcopenshell 없이도 동작
    IFC_OK, IFC_ERR = False, str(_e)


# ═══════════════════════════════════════════════════════════════════════════
# 0. 상수 · 공종 체계
# ═══════════════════════════════════════════════════════════════════════════
# (key, 판정용 IFC 클래스, 시공 위계 rank, 한글명, 기본 공기[작업일])
CATEGORIES = [
    ("FOUNDATION", ("IfcFooting", "IfcPile", "IfcDeepFoundation"), 0, "기초", 4),
    ("COLUMN", ("IfcColumn",), 1, "기둥", 2),
    ("WALL", ("IfcWall",), 2, "벽", 2),
    ("BEAM", ("IfcBeam",), 3, "보", 2),
    ("SLAB", ("IfcSlab",), 4, "슬래브", 3),
    ("STAIR", ("IfcStair", "IfcStairFlight", "IfcRamp", "IfcRampFlight"), 5, "계단·램프", 2),
    ("MEP", ("IfcDistributionElement", "IfcFlowSegment", "IfcFlowTerminal", "IfcFlowFitting"), 5, "MEP 설비", 1),
    ("ENVELOPE", ("IfcCurtainWall", "IfcPlate", "IfcMember", "IfcRoof"), 6, "지붕·외장", 2),
    ("OPENING", ("IfcDoor", "IfcWindow"), 7, "문·창호", 1),
    ("FINISH", ("IfcCovering", "IfcRailing"), 8, "마감·난간", 1),
    ("FURNITURE", ("IfcFurnishingElement",), 9, "가구·집기", 1),
    ("OTHER", (), 5, "기타", 1),
]
CAT_RANK = {c[0]: c[2] for c in CATEGORIES}
CAT_LABEL = {c[0]: c[3] for c in CATEGORIES}
CAT_DAYS = {c[0]: c[4] for c in CATEGORIES}
STRUCTURAL_MAX_RANK = 4  # 기초~슬래브

EDGE_WEIGHT = {"ifc_fill": 3.0, "support": 3.0, "containment": 2.5, "embedding": 2.0,
               "gravity": 1.5, "precedence": 1.0}
REL_KO = {"support": "지지(상하)", "ifc_fill": "IFC 개구부 채움", "containment": "포함",
          "embedding": "매립", "precedence": "접촉·공종 위계", "gravity": "중력 지지 추정"}
# 엣지별 기하(공간) 관계 — 'Spatial' 열이 없는 이전 세션 결과용 대체 규칙
SPATIAL_FALLBACK = {"containment": "containment", "embedding": "embedding", "ifc_fill": "embedding",
                    "gravity": "non-contact", "support": "contact", "precedence": "contact"}
VOL_EPS = 1e-4      # m³, 이보다 작은 겹침은 접촉으로 간주
EMBED_RATIO = 0.02  # 작은 부재 부피 대비 겹침 비율 임계값

SCHED_COLS = ["Layer", "GlobalId", "Duration_d", "Start_day", "Finish_day",
              "Layer_start_day", "Layer_finish_day", "P_support", "P_impact", "P_critical"]
PRINCIPLES = {"지지값 우선 (support-first)": "support",
              "영향값 우선 (impact-first)": "impact",
              "임계경로 우선 (critical-chain)": "critical"}
WEEKMASKS = {"주 5일 (월~금)": "1111100", "주 6일 (월~토)": "1111110", "주 7일 (무휴)": "1111111"}


# ═══════════════════════════════════════════════════════════════════════════
# 1. 공통 유틸 · 캘린더
# ═══════════════════════════════════════════════════════════════════════════
class RunLogger:
    """파이프라인 로그를 모으면서 st.status 컨테이너에도 실시간 출력."""

    def __init__(self, sink=None, bar=None):
        self.lines, self.sink, self.bar, self.t0 = [], sink, bar, time.perf_counter()

    def __call__(self, msg: str):
        line = f"[{time.perf_counter() - self.t0:6.2f}s] {msg}"
        self.lines.append(line)
        if self.sink is not None:
            self.sink.write(line)

    def progress(self, frac: float, text: str | None = None):
        if self.bar is not None:
            self.bar.progress(float(min(max(frac, 0.0), 1.0)), text=text)


def make_calendar(start: date, weekmask: str, holidays: list[str]) -> dict:
    hol = np.array(sorted(set(holidays)), dtype="datetime64[D]") if holidays else np.array([], dtype="datetime64[D]")
    start64 = np.busday_offset(np.datetime64(start, "D"), 0, roll="forward", weekmask=weekmask, holidays=hol)
    return {"start": start64, "wm": weekmask, "hol": hol, "wpw": weekmask.count("1")}


def day_to_date(cal: dict, days):
    """작업일 인덱스(0=착공일) → 달력 날짜."""
    arr = np.busday_offset(cal["start"], np.asarray(days, dtype=np.int64), roll="forward",
                           weekmask=cal["wm"], holidays=cal["hol"])
    return pd.to_datetime(arr)


def date_to_day(cal: dict, d) -> int:
    n = np.busday_count(cal["start"], np.datetime64(d, "D"), weekmask=cal["wm"], holidays=cal["hol"])
    return int(max(0, n))


_WIN_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def safe_filename_stem(source: str | None, max_len: int = 60, fallback: str = "model") -> str:
    """모델 소스 이름 → 파일명 접두사로 안전한 문자열.

    'SampleBuilding.ifc' → 'SampleBuilding', '샘플 모델 (5F RC)' → '샘플_모델_5F_RC'
    - IFC 계열 확장자 제거, 유니코드 NFC 정규화(macOS 한글 자모 분리 방지)
    - 한글·영문·숫자·'-' 외 문자(공백, 괄호, 경로 구분자, 따옴표 등)는 '_'로 치환 후 연속 '_' 축약
    - Windows 예약어(CON, NUL …) 회피, 길이 제한, 비면 fallback 사용
    """
    s = unicodedata.normalize("NFC", str(source or "")).strip()
    s = re.sub(r"\.(ifc|ifczip|ifcxml|ifcjson)$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\-]+", "_", s)          # \w는 유니코드 기준이라 한글 유지
    s = re.sub(r"_+", "_", s).strip("._-")
    s = s[:max_len].rstrip("._-")
    if not s:
        return fallback
    if s.lower() in _WIN_RESERVED:
        s = f"{s}_{fallback}"
    return s


def export_name(proj: dict, suffix: str) -> str:
    """현재 모델 기준 내보내기 파일명: '[모델명]_[suffix]'."""
    return f"{safe_filename_stem(proj.get('source'))}_{suffix}"


def _rand_guid(rng: random.Random) -> str:
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    n, out = rng.getrandbits(128), []
    for _ in range(22):
        out.append(chars[n % 64])
        n //= 64
    return "".join(out)


def _finalize_components(rows: list[dict], storeys: pd.DataFrame) -> pd.DataFrame:
    comps = pd.DataFrame(rows)
    # GlobalId 중복 방지
    dup = comps["GlobalId"].duplicated(keep="first")
    if dup.any():
        comps.loc[dup, "GlobalId"] = comps.loc[dup, "GlobalId"] + "_" + comps.loc[dup, "ExpressId"].astype(str)
    comps["Rank"] = comps["Category"].map(CAT_RANK).fillna(5).astype(int)
    comps["Label"] = comps["Category"].map(CAT_LABEL).fillna("기타")
    idx_map = dict(zip(storeys["Storey"], storeys["StoreyIdx"])) if len(storeys) else {}
    comps["StoreyIdx"] = comps["Storey"].map(idx_map).fillna(999).astype(int)
    return comps.set_index("GlobalId")


# ═══════════════════════════════════════════════════════════════════════════
# 2. IFC 추출 (tsg_builder 전처리부)
# ═══════════════════════════════════════════════════════════════════════════
def _classify(el) -> str:
    for key, classes, *_ in CATEGORIES:
        for c in classes:
            try:
                if el.is_a(c):
                    return key
            except Exception:
                continue
    return "OTHER"


def _storey_of(el):
    obj, hops = el, 0
    while obj is not None and hops < 12:
        c = ifc_el.get_container(obj)
        if c is not None:
            cur = c
            while cur is not None and not cur.is_a("IfcBuildingStorey") and hops < 12:
                cur = ifc_el.get_aggregate(cur)
                hops += 1
            return cur
        obj = ifc_el.get_aggregate(obj)
        hops += 1
    return None


def _is_decomposed(el) -> bool:
    for rel in getattr(el, "IsDecomposedBy", None) or []:
        if getattr(rel, "RelatedObjects", None):
            return True
    return False


def _geom_settings():
    s = ifcopenshell.geom.settings()
    try:
        s.set("use-world-coords", True)          # ifcopenshell 0.8+
    except Exception:
        try:
            s.set(s.USE_WORLD_COORDS, True)       # ifcopenshell 0.7
        except Exception:
            pass
    return s


def extract_from_ifc(path: str, log: RunLogger):
    model = ifcopenshell.open(path)
    scale = ifc_unit.calculate_unit_scale(model)
    log(f"IFC 로드 완료 · 스키마 {model.schema} · 길이단위 배율 {scale:g} m")

    # 층 정보
    st_rows = []
    for s in model.by_type("IfcBuildingStorey"):
        elev = getattr(s, "Elevation", None)
        st_rows.append({"Storey": s.Name or f"Storey#{s.id()}",
                        "Elevation": float(elev) * scale if elev is not None else np.nan})
    storeys = pd.DataFrame(st_rows, columns=["Storey", "Elevation"]).drop_duplicates("Storey")
    log(f"층(IfcBuildingStorey) {len(storeys)}개")

    # 대상 부재 선별
    elements = []
    for el in model.by_type("IfcElement"):
        if el.is_a("IfcFeatureElement") or el.is_a("IfcVirtualElement"):
            continue
        if not getattr(el, "Representation", None) or _is_decomposed(el):
            continue
        elements.append(el)
    log(f"시공 대상 부재 {len(elements)}개 선별 (개구부·가상요소·상위 집합체 제외)")
    if not elements:
        raise ValueError("형상이 있는 IfcElement를 찾지 못했습니다.")

    # 형상 → AABB (월드 좌표, m)
    bbox, settings, total = {}, _geom_settings(), len(elements)
    try:
        threads = max(1, (os.cpu_count() or 2) - 1)
        it = ifcopenshell.geom.iterator(settings, model, threads, include=elements)
        if it.initialize():
            n = 0
            while True:
                sh = it.get()
                v = np.asarray(sh.geometry.verts, dtype=float).reshape(-1, 3)
                if len(v):
                    bbox[int(sh.id)] = np.concatenate([v.min(0), v.max(0)])
                n += 1
                if n % 200 == 0:
                    log.progress(0.05 + 0.55 * n / total, f"형상 처리 {n}/{total}")
                if not it.next():
                    break
    except Exception as e:
        log(f"⚠️ geom.iterator 실패({e}) → 부재별 create_shape로 대체")
    missing = [el for el in elements if el.id() not in bbox]
    for k, el in enumerate(missing):
        try:
            sh = ifcopenshell.geom.create_shape(settings, el)
            v = np.asarray(sh.geometry.verts, dtype=float).reshape(-1, 3)
            if len(v):
                bbox[el.id()] = np.concatenate([v.min(0), v.max(0)])
        except Exception:
            pass
        if k % 200 == 0 and missing:
            log.progress(0.6 * (k + 1) / len(missing), f"형상 보완 {k + 1}/{len(missing)}")
    log(f"형상 추출 {len(bbox)}/{total}개 성공")

    rows, no_geom = [], 0
    for el in elements:
        b = bbox.get(el.id())
        if b is None:  # 형상 실패 → 배치점 기준 소형 박스
            try:
                p = ifc_pl.get_local_placement(el.ObjectPlacement)[:3, 3] * scale
                b = np.concatenate([p - 0.05, p + 0.05])
                no_geom += 1
            except Exception:
                continue
        sto = _storey_of(el)
        rows.append({"GlobalId": el.GlobalId, "ExpressId": el.id(), "Name": el.Name or "",
                     "IfcType": el.is_a(), "Category": _classify(el),
                     "Storey": (sto.Name or f"Storey#{sto.id()}") if sto is not None else None,
                     "xmin": b[0], "ymin": b[1], "zmin": b[2], "xmax": b[3], "ymax": b[4], "zmax": b[5]})
    if no_geom:
        log(f"⚠️ 형상 없는 부재 {no_geom}개는 배치점 기준 0.1 m 박스로 근사")
    df = pd.DataFrame(rows)

    # 층 표고 보정(누락 시 부재 zmin 중앙값) · 층 미지정 부재는 표고 기준 배정
    if len(storeys):
        med = df.groupby("Storey")["zmin"].median()
        storeys["Elevation"] = storeys.apply(
            lambda r: r["Elevation"] if pd.notna(r["Elevation"]) else med.get(r["Storey"], 0.0), axis=1)
        storeys = storeys.sort_values("Elevation").reset_index(drop=True)
        storeys["StoreyIdx"] = np.arange(len(storeys))
        elev = storeys["Elevation"].to_numpy()
        miss = df["Storey"].isna()
        if miss.any():
            pos = np.clip(np.searchsorted(elev, df.loc[miss, "zmin"].to_numpy() + 0.1, side="right") - 1, 0, len(elev) - 1)
            df.loc[miss, "Storey"] = storeys["Storey"].to_numpy()[pos]
            log(f"층 미지정 부재 {int(miss.sum())}개 → 표고 기준으로 층 배정")
    else:
        df["Storey"] = df["Storey"].fillna("(층 없음)")
        storeys = pd.DataFrame({"Storey": ["(층 없음)"], "Elevation": [0.0], "StoreyIdx": [0]})

    # IFC 명시 관계: 호스트(벽 등) → 개구부 → 문/창 채움
    explicit = []
    for rel in model.by_type("IfcRelFillsElement"):
        try:
            filler = rel.RelatedBuildingElement
            for v in rel.RelatingOpeningElement.VoidsElements or []:
                explicit.append((v.RelatingBuildingElement.GlobalId, filler.GlobalId))
        except Exception:
            continue
    log(f"IfcRelFillsElement 기반 매립 관계 {len(explicit)}건")
    return _finalize_components(df.to_dict("records"), storeys), storeys, explicit


# ═══════════════════════════════════════════════════════════════════════════
# 3. 샘플(데모) 모델 — ifcopenshell 없이 UI·알고리즘 체험용
# ═══════════════════════════════════════════════════════════════════════════
def build_demo_model(floors: int = 5, gx: int = 5, gy: int = 4, bay: float = 6.0, H: float = 3.6, seed: int = 7):
    rng, rows, explicit, eid = random.Random(seed), [], [], [100]
    st_list = [("기초", -1.0)] + [(f"{f + 1}F", f * H) for f in range(floors)] + [("RF", floors * H)]
    storeys = pd.DataFrame(st_list, columns=["Storey", "Elevation"])
    storeys["StoreyIdx"] = np.arange(len(storeys))

    def add(ifc, cat, name, storey, b):
        g = _rand_guid(rng)
        eid[0] += 1
        rows.append({"GlobalId": g, "ExpressId": eid[0], "Name": name, "IfcType": ifc, "Category": cat,
                     "Storey": storey, "xmin": b[0], "ymin": b[1], "zmin": b[2],
                     "xmax": b[3], "ymax": b[4], "zmax": b[5]})
        return g

    X, Y = [i * bay for i in range(gx)], [j * bay for j in range(gy)]
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            add("IfcFooting", "FOUNDATION", f"독립기초 F{i}{j}", "기초", (x - .8, y - .8, -1.0, x + .8, y + .8, 0.0))
    for lv in range(floors + 1):
        z, sn = lv * H, st_list[lv + 1][0]
        for i in range(gx - 1):
            for j in range(gy - 1):
                add("IfcSlab", "SLAB", f"슬래브 {sn}-{i}{j}", sn, (X[i], Y[j], z, X[i + 1], Y[j + 1], z + .2))
    for f in range(floors):
        z0, sn = f * H, st_list[f + 1][0]
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                add("IfcColumn", "COLUMN", f"기둥 {sn}-C{i}{j}", sn, (x - .25, y - .25, z0 + .2, x + .25, y + .25, z0 + H))
        for y in Y:
            for i in range(gx - 1):
                add("IfcBeam", "BEAM", f"보 {sn}-X{i}", sn, (X[i] + .25, y - .2, z0 + H - .6, X[i + 1] - .25, y + .2, z0 + H))
        for x in X:
            for j in range(gy - 1):
                add("IfcBeam", "BEAM", f"보 {sn}-Y{j}", sn, (x - .2, Y[j] + .25, z0 + H - .6, x + .2, Y[j + 1] - .25, z0 + H))
        for y in (Y[0], Y[-1]):
            for i in range(gx - 1):
                w = add("IfcWall", "WALL", f"외벽 {sn}-{'S' if y == Y[0] else 'N'}{i}", sn,
                        (X[i] + .25, y - .1, z0 + .2, X[i + 1] - .25, y + .1, z0 + H - .6))
                xm = (X[i] + X[i + 1]) / 2
                if y == Y[0] and i == 1:
                    d = add("IfcDoor", "OPENING", f"출입문 {sn}", sn, (xm - .5, y - .1, z0 + .2, xm + .5, y + .1, z0 + 2.3))
                    explicit.append((w, d))
                elif y == Y[-1]:
                    win = add("IfcWindow", "OPENING", f"창호 {sn}-{i}", sn, (xm - .75, y - .1, z0 + 1.0, xm + .75, y + .1, z0 + 2.2))
                    explicit.append((w, win))
        for x in (X[0], X[-1]):
            for j in range(gy - 1):
                add("IfcWall", "WALL", f"외벽 {sn}-{'W' if x == X[0] else 'E'}{j}", sn,
                    (x - .1, Y[j] + .25, z0 + .2, x + .1, Y[j + 1] - .25, z0 + H - .6))
        for j in range(gy - 1):  # 덕트(천장 내부) — 배관과 교차 → 동일 레이어 금지 충돌
            ym = (Y[j] + Y[j + 1]) / 2
            for i in range(gx - 1):
                add("IfcDuctSegment", "MEP", f"덕트 {sn}-{j}{i}", sn, (X[i], ym - .3, z0 + 2.55, X[i + 1], ym + .3, z0 + 2.85))
        for i in range(gx - 1):
            xm = (X[i] + X[i + 1]) / 2
            for j in range(gy - 1):
                add("IfcPipeSegment", "MEP", f"배관 {sn}-{i}{j}", sn, (xm - .1, Y[j], z0 + 2.6, xm + .1, Y[j + 1], z0 + 2.8))
    return _finalize_components(rows, storeys), storeys, explicit


# ═══════════════════════════════════════════════════════════════════════════
# 4. TSG 구축 (tsg_builder)
# ═══════════════════════════════════════════════════════════════════════════
def _broad_phase(B: np.ndarray, tol: float):
    """x축 sweep-and-prune: 허용오차 내에서 AABB가 맞닿거나 겹치는 쌍만 추림."""
    order = np.argsort(B[:, 0], kind="stable")
    Bs = B[order]
    xs = Bs[:, 0]
    I, J = [], []
    for a in range(len(Bs)):
        hi = np.searchsorted(xs, Bs[a, 3] + tol, side="right")
        if hi <= a + 1:
            continue
        c = Bs[a + 1:hi]
        m = ((c[:, 1] <= Bs[a, 4] + tol) & (c[:, 4] >= Bs[a, 1] - tol) &
             (c[:, 2] <= Bs[a, 5] + tol) & (c[:, 5] >= Bs[a, 2] - tol))
        idx = np.nonzero(m)[0]
        if len(idx):
            I.append(np.full(len(idx), a))
            J.append(idx + a + 1)
    if not I:
        return np.array([], dtype=int), np.array([], dtype=int)
    return order[np.concatenate(I)], order[np.concatenate(J)]


def build_tsg(comps: pd.DataFrame, explicit: list, tol: float, log: RunLogger) -> dict:
    guids = comps.index.to_numpy()
    n = len(guids)
    B = comps[["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"]].to_numpy(float)
    rank = comps["Rank"].to_numpy()

    I, J = _broad_phase(B, tol)
    log.progress(0.7, "공간 관계 판정")
    log(f"근접 부재쌍(허용오차 {tol} m) {len(I):,}쌍 탐색")

    # ── 공간 관계 판정: 포함 / 매립 / 접촉
    ext = np.clip(B[:, 3:] - B[:, :3], 1e-6, None)
    vol_b = ext.prod(1)
    lo, hi = np.maximum(B[I, :3], B[J, :3]), np.minimum(B[I, 3:], B[J, 3:])
    ov = hi - lo
    ovp = np.clip(ov, 0, None)
    vol = ovp.prod(1)
    ratio = vol / np.minimum(vol_b[I], vol_b[J])
    in_ij = np.all(B[I, :3] >= B[J, :3] - tol, 1) & np.all(B[I, 3:] <= B[J, 3:] + tol, 1)
    in_ji = np.all(B[J, :3] >= B[I, :3] - tol, 1) & np.all(B[J, 3:] <= B[I, 3:] + tol, 1)
    rel = np.full(len(I), "contact", dtype=object)
    emb = (vol > VOL_EPS) & (ratio > EMBED_RATIO)
    rel[emb] = "embedding"
    rel[emb & (in_ij | in_ji)] = "containment"

    # 접촉면적 비율(정량 서술자): 법선축 = 겹침이 가장 얇은 축
    normal = np.argmin(np.abs(ov), 1) if len(I) else np.array([], dtype=int)
    contact_ratio = np.zeros(len(I))
    for a in range(3):
        m = normal == a
        if not m.any():
            continue
        b, c = [k for k in range(3) if k != a]
        area = ovp[m, b] * ovp[m, c]
        face = np.minimum(ext[I[m], b] * ext[I[m], c], ext[J[m], b] * ext[J[m], c])
        contact_ratio[m] = np.clip(area / np.maximum(face, 1e-9), 0, 1)

    # ── 속성 기반 방향 보정
    xy = ovp[:, 0] * ovp[:, 1]
    foot = np.minimum(ext[I, 0] * ext[I, 1], ext[J, 0] * ext[J, 1])
    stack_ok = (xy >= 0.01) | (xy / np.maximum(foot, 1e-9) >= 0.2)
    ztol = max(tol, 0.05)
    i_below = stack_ok & (B[I, 5] <= B[J, 2] + ztol)                 # (1) 상하 지지: 아래 → 위
    j_below = stack_ok & (B[J, 5] <= B[I, 2] + ztol) & ~i_below
    rest = ~(i_below | j_below)
    ri, rj = rank[I], rank[J]
    lt, gt = rest & (ri < rj), rest & (ri > rj)                         # (2) 공종 위계
    conflict = rest & (ri == rj) & (rel != "contact")                   # (3) 동일 위계 겹침 → 동시 시공 금지

    src = np.full(len(I), -1)
    dst = np.full(len(I), -1)
    etype = np.full(len(I), "", dtype=object)
    src[i_below], dst[i_below], etype[i_below] = I[i_below], J[i_below], "support"
    src[j_below], dst[j_below], etype[j_below] = J[j_below], I[j_below], "support"
    rel_dir = np.where(rel == "contact", "precedence", rel)
    src[lt], dst[lt], etype[lt] = I[lt], J[lt], rel_dir[lt]
    src[gt], dst[gt], etype[gt] = J[gt], I[gt], rel_dir[gt]

    G = nx.DiGraph()
    G.add_nodes_from(guids)
    has = src >= 0
    for s, d, e, cr, sp in zip(src[has], dst[has], etype[has], contact_ratio[has], rel[has]):
        G.add_edge(guids[s], guids[d], relation=e, spatial=sp,
                   weight=EDGE_WEIGHT[e] + float(cr), contact_ratio=round(float(cr), 3))
    pair_rel = {frozenset((guids[a], guids[b])): r for a, b, r in zip(I, J, rel)}  # 기하 관계 조회용

    # (4) IFC 명시 관계(개구부 채움)는 기하 판정보다 우선
    n_fill = 0
    for h, f in explicit:
        if h in G and f in G and h != f:
            if G.has_edge(f, h):
                G.remove_edge(f, h)
            G.add_edge(h, f, relation="ifc_fill", spatial=pair_rel.get(frozenset((h, f)), "embedding"),
                       weight=EDGE_WEIGHT["ifc_fill"], contact_ratio=1.0)
            n_fill += 1

    # (5) 상향식 원칙: 선행 부재가 없는 비기초 부재 → 바로 아래 구조 부재에 매달기(중력 지지 추정)
    struct = np.nonzero(rank <= STRUCTURAL_MAX_RANK)[0]
    n_grav = 0
    if len(struct):
        Bs = B[struct]
        for k, g in enumerate(guids):
            if rank[k] == 0 or G.in_degree(g) > 0:
                continue
            m = ((Bs[:, 0] < B[k, 3]) & (Bs[:, 3] > B[k, 0]) & (Bs[:, 1] < B[k, 4]) & (Bs[:, 4] > B[k, 1]) &
                 (Bs[:, 5] <= B[k, 2] + ztol) & (struct != k))
            if m.any():
                cand = np.nonzero(m)[0]
                s = struct[cand[np.argmax(Bs[cand, 5])]]
                if not G.has_edge(g, guids[s]):
                    G.add_edge(guids[s], g, relation="gravity", spatial="non-contact",
                               weight=EDGE_WEIGHT["gravity"], contact_ratio=0.0)
                    n_grav += 1

    # (6) 순환 제거: 강연결요소 안에서 가장 약한 엣지부터 제거 (Tarjan SCC)
    removed = 0
    for scc in list(nx.strongly_connected_components(G)):
        if len(scc) < 2:
            continue
        sub = G.subgraph(scc).copy()
        while True:
            try:
                cyc = nx.find_cycle(sub)
            except nx.NetworkXNoCycle:
                break
            u, v = min(cyc, key=lambda e: sub.edges[e[0], e[1]]["weight"])[:2]
            sub.remove_edge(u, v)
            G.remove_edge(u, v)
            removed += 1

    # 동일 레이어 금지 충돌 쌍(의존 엣지가 없는 겹침)
    conf_adj: dict[str, set] = {}
    n_conf = 0
    for a, b in zip(I[conflict], J[conflict]):
        ga, gb = guids[a], guids[b]
        if G.has_edge(ga, gb) or G.has_edge(gb, ga):
            continue
        conf_adj.setdefault(ga, set()).add(gb)
        conf_adj.setdefault(gb, set()).add(ga)
        n_conf += 1

    edges = pd.DataFrame([(u, v, d["relation"], d.get("spatial", SPATIAL_FALLBACK.get(d["relation"], "contact")),
                           round(d["weight"], 3), d["contact_ratio"])
                          for u, v, d in G.edges(data=True)],
                         columns=["Source", "Target", "Relation", "Spatial", "Weight", "ContactRatio"])
    rel_counts = pd.Series(rel).value_counts().to_dict() if len(rel) else {}
    stats = {"nodes": n, "pairs": int(len(I)), "edges": G.number_of_edges(), "fills": n_fill,
             "gravity": n_grav, "cycles_removed": removed, "conflicts": n_conf,
             "CI": round(2 * len(I) / max(n, 1), 2), "rel_counts": rel_counts,
             "edge_counts": edges["Relation"].value_counts().to_dict()}
    log(f"공간 관계: 접촉 {rel_counts.get('contact', 0):,} · 매립 {rel_counts.get('embedding', 0):,} · "
        f"포함 {rel_counts.get('containment', 0):,} · 복잡도지수 CI={stats['CI']}")
    log(f"의존 엣지 {stats['edges']:,}개 (IFC 채움 {n_fill} · 중력 지지 추정 {n_grav}) · "
        f"순환 제거 {removed} · 동시시공 금지 쌍 {n_conf}")
    return {"G": G, "edges": edges, "conf_adj": conf_adj, "stats": stats}


# ═══════════════════════════════════════════════════════════════════════════
# 5. DLSA 스케줄러 (dlsa_scheduler)
# ═══════════════════════════════════════════════════════════════════════════
def critical_path(G: nx.DiGraph, dur: dict) -> dict:
    cp = {}
    for n in reversed(list(nx.topological_sort(G))):
        cp[n] = dur.get(n, 1) + max((cp[s] for s in G.successors(n)), default=0)
    return cp


def run_dlsa(G, comps, durations, n_max, principle, release=None, start_day=0, start_layer=1,
             frozen=frozenset(), conf_adj=None):
    """Algorithm 1 (DLSA) + 착수 가능일(release) 제약 + 동일 레이어 충돌 회피."""
    release, conf_adj, n_max = release or {}, conf_adj or {}, max(1, int(n_max))
    remaining = set(G.nodes) - set(frozen)
    indeg = {v: sum(1 for p in G.predecessors(v) if p in remaining) for v in remaining}
    ready = {v for v, d in indeg.items() if d == 0}
    cp = critical_path(G, durations)
    sidx, zmin = comps["StoreyIdx"].to_dict(), comps["zmin"].to_dict()
    rows, stats = [], {"conflict_deferrals": 0, "idle_days": 0}
    t, k = int(start_day), int(start_layer)

    while remaining:
        if not ready:
            raise RuntimeError("의존성 그래프에 순환이 남아 있습니다.")
        avail = [v for v in ready if release.get(v, 0) <= t]
        if not avail:                      # 남은 후보가 모두 착수 불가 → 가장 이른 가능일로 이동
            nt = min(release.get(v, 0) for v in ready)
            stats["idle_days"] += nt - t
            t = nt
            continue
        imp = {v: sum(1 for s in G.successors(v) if indeg.get(s) == 1) for v in avail}   # I(E)
        sup = {v: G.out_degree(v) for v in avail}                                          # S(E)

        def key(v):
            head = {"impact": (imp[v], sup[v], cp[v]), "critical": (cp[v], sup[v], imp[v])}.get(
                principle, (sup[v], imp[v], cp[v]))
            return head + (-sidx.get(v, 999), -zmin.get(v, 0.0), v)

        avail.sort(key=key, reverse=True)
        chosen, cs = [], set()
        for v in avail:
            if len(chosen) >= n_max:
                break
            nb = conf_adj.get(v)
            if nb and not nb.isdisjoint(cs):   # 선시공 이상 탐지: 같은 레이어 공간 간섭 → 다음 레이어로
                stats["conflict_deferrals"] += 1
                continue
            chosen.append(v)
            cs.add(v)
        ldur = max(durations[v] for v in chosen)
        for v in chosen:
            rows.append((k, v, durations[v], t, t + durations[v], t, t + ldur, sup[v], imp[v], cp[v]))
        for v in chosen:
            remaining.discard(v)
            ready.discard(v)
            for s in G.successors(v):
                if s in remaining:
                    indeg[s] -= 1
                    if indeg[s] == 0:
                        ready.add(s)
        t += ldur
        k += 1
    stats["layers"] = k - start_layer
    return pd.DataFrame(rows, columns=SCHED_COLS), stats


def component_durations(proj: dict) -> dict:
    base = proj["comps"]["Category"].map(proj["cat_days"]).fillna(1).astype(int).clip(lower=1)
    dur = base.to_dict()
    for g, d in proj["extra"].items():
        dur[g] = max(1, dur[g] + int(d))
    return dur


def reschedule(proj: dict, status_day: int, cur: pd.DataFrame | None = None):
    """기준일 이전 착수 레이어는 동결하고, 잔여 서브그래프만 DLSA로 재계층화."""
    cur = proj["schedule"] if cur is None else cur
    frozen_df = cur[cur["Layer_start_day"] < status_day]
    if len(frozen_df):
        resume = int(max(status_day, frozen_df["Layer_finish_day"].max(), frozen_df["Finish_day"].max()))
        start_layer = int(frozen_df["Layer"].max()) + 1
    else:
        resume, start_layer = int(status_day), 1
    res, stats = run_dlsa(proj["G"], proj["comps"], component_durations(proj), proj["n_max"], proj["principle"],
                          release=proj["release"], start_day=resume, start_layer=start_layer,
                          frozen=frozenset(frozen_df["GlobalId"]), conf_adj=proj["conf_adj"])
    new = pd.concat([frozen_df, res], ignore_index=True).sort_values(["Layer", "GlobalId"]).reset_index(drop=True)
    stats["frozen_layers"] = int(frozen_df["Layer"].nunique())
    stats["resume_day"] = resume
    return new, stats


def schedule_summary(sched: pd.DataFrame, cal: dict) -> dict:
    if not len(sched):
        return {"total": 0, "layers": 0, "act": 0.0, "finish": None}
    total = int(sched["Finish_day"].max())
    layers = int(sched["Layer"].nunique())
    return {"total": total, "layers": layers, "act": round(len(sched) / layers, 2),
            "finish": day_to_date(cal, [max(total - 1, 0)])[0].date()}


# ═══════════════════════════════════════════════════════════════════════════
# 6. 자연어 돌발 변수 파서
# ═══════════════════════════════════════════════════════════════════════════
PARTICLES = "에서|으로|부터|까지|을|를|이|가|은|는|의|도|와|과|만|에|로|들"
TARGET_KEYWORDS = [
    (("cat", "FOUNDATION"), ["기초", "말뚝", "footing", "foundation", "pile"]),
    (("cat", "COLUMN"), ["기둥", "column"]),
    (("cat", "WALL"), ["벽체", "옹벽", "벽", "wall"]),
    (("cat", "BEAM"), ["거더", "보", "beam", "girder"]),
    (("cat", "SLAB"), ["슬래브", "슬라브", "바닥판", "slab"]),
    (("cat", "STAIR"), ["계단", "램프", "stair", "ramp"]),
    (("cat", "MEP"), ["배관", "덕트", "설비", "전기", "소방", "스프링클러", "케이블", "mep", "pipe", "duct"]),
    (("cat", "ENVELOPE"), ["커튼월", "지붕", "외장", "패널", "curtain", "roof", "facade"]),
    (("cat", "OPENING"), ["창호"]),
    (("cls", "IfcDoor"), ["도어", "문", "door"]),
    (("cls", "IfcWindow"), ["창문", "창", "window"]),
    (("cat", "FINISH"), ["마감", "천장", "난간", "covering", "ceiling", "railing"]),
    (("cat", "FURNITURE"), ["가구", "furniture"]),
]
EQUIPMENT = [
    (r"크레인|양중|crane", ["COLUMN", "BEAM", "SLAB", "ENVELOPE"], "타워크레인/양중"),
    (r"펌프카|레미콘|타설|콘크리트|concrete|pump", ["FOUNDATION", "COLUMN", "WALL", "BEAM", "SLAB"], "콘크리트 타설"),
    (r"호이스트|리프트|hoist|lift", ["MEP", "OPENING", "FINISH", "ENVELOPE"], "호이스트/리프트"),
]
GLOBAL_RE = r"우천|폭우|폭설|강풍|태풍|한파|폭염|미세먼지|작업\s*중지|공사\s*중단|전면\s*중단|현장\s*폐쇄|셧다운|파업|shutdown|rain|storm|strike"
EXTEND_RE = r"연장|더\s*걸|추가\s*소요|늘어나|길어|재시공|하자|보수|extend|longer|rework"
DELAY_RE = r"지연|늦|입고|반입|미착|보류|홀드|연기|미뤄|delay|late|hold|postpone|고장|불가"
NMAX_RE = r"동시|병렬|최대\s*작업|작업조|크루|crew|parallel|n_?max|투입\s*인력|인력"
KOR_DAYS = {"하루": 1, "이틀": 2, "사흘": 3, "나흘": 4, "닷새": 5, "엿새": 6, "열흘": 10, "보름": 15}
HELP_TEXT = """**입력 예시** (자연어로 자유롭게 써도 됩니다)
- `2층 벽 자재 입고 3일 지연` → 해당 부재 착수를 3작업일 뒤로
- `#1234 부재 이틀 늦어짐` · `레이어 12 하루 지연` · `"외벽 3F-N2" 2일 보류`
- `3층 기둥 시공 2일 연장` → 공기(작업기간) 연장
- `타워크레인 고장 4일` → 양중 공종(기둥·보·슬래브·외장) 4일간 착수 불가
- `우천으로 2일 작업 중지` → 전 공종 중지 + 진행 중 작업 연장
- `2026-11-20 기준 폭우 3일 공사 중단` → 기준일을 문장 속 날짜로 지정
- `인력 부족으로 동시 작업 10개로 축소` → N_max 변경
- `현황` · `되돌리기` · `초기화` · `도움말`

기준일(사이드바 '진도 기준일') 이전에 착수한 레이어는 동결되고, 이후 잔여 부재만 DLSA로 다시 계층화합니다."""


def _kw_regex(w: str) -> str:
    if re.fullmatch(r"[가-힣]", w):  # 한 글자 한국어(보·벽·문·창)는 조사만 허용해 오탐 방지
        return rf"(?<![가-힣]){w}(?=$|[^가-힣]|(?:{PARTICLES})(?=$|[^가-힣]))"
    if re.fullmatch(r"[a-z_ ]+", w):
        return rf"\b{w}s?\b"
    return re.escape(w)


def _extract_days(t: str, wpw: int):
    total, found = 0.0, False
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:개월|months?)", t):
        total += float(m.group(1)) * wpw * 4.345
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:주일|주간|주|weeks?|wks?)(?![가-힣])", t):
        total += float(m.group(1)) * wpw
        found = True
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:작업일|영업일|일간|일|days?|d\b)", t):
        total += float(m.group(1))
        found = True
    if re.search(r"일주일|한\s*주", t):
        total += wpw
        found = True
    for w, v in KOR_DAYS.items():
        if w in t:
            total += v
            found = True
    return int(math.ceil(total)) if found and total > 0 else None


def _extract_status_date(t: str, cal: dict):
    start = pd.Timestamp(cal["start"]).date()
    m = re.search(r"(20\d{2})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})일?", t)
    d = None
    if m:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        m = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", t)
        if m:
            d = date(start.year, int(m.group(1)), int(m.group(2)))
            if d < start:
                d = date(start.year + 1, d.month, d.day)
        elif "오늘" in t or "today" in t:
            d = date.today()
    if m:
        t = t[:m.start()] + " " + t[m.end():]
    return (date_to_day(cal, d) if d else None), d, t


def _match_storeys(t: str, storeys: pd.DataFrame):
    names, found = list(storeys["Storey"]), []
    above = storeys[storeys["Elevation"] >= -0.3].sort_values("Elevation")["Storey"].tolist()
    below = storeys[storeys["Elevation"] < -0.3].sort_values("Elevation", ascending=False)["Storey"].tolist()

    def by_number(n: int, basement: bool):
        hits = []
        for s in names:
            sl = str(s).lower()
            is_b = bool(re.search(r"^b|지하|basement", sl))
            if is_b == basement and re.search(rf"(?<!\d){n}(?!\d)", sl):
                hits.append(s)
        if not hits:
            pool = below if basement else above
            if 1 <= n <= len(pool):
                hits = [pool[n - 1]]
        return hits

    for m in re.finditer(r"(?:지하\s*|\bb)(\d+)\s*(?:층|f\b)?", t):
        found += by_number(int(m.group(1)), True)
    t2 = re.sub(r"(?:지하\s*|\bb)(\d+)\s*(?:층|f\b)?", " ", t)
    for m in re.finditer(r"(\d+)\s*[~\-]\s*(\d+)\s*(?:층|f\b)", t2):
        for n in range(int(m.group(1)), int(m.group(2)) + 1):
            found += by_number(n, False)
    t2 = re.sub(r"(\d+)\s*[~\-]\s*(\d+)\s*(?:층|f\b)", " ", t2)
    for m in re.finditer(r"(?<![\d.])(\d+)\s*(?:층|f\b)|(?:level|레벨)\s*(\d+)", t2):
        found += by_number(int(m.group(1) or m.group(2)), False)
    for s in names:  # 'RF', '지붕층' 등 이름 그대로 언급
        if len(str(s)) >= 2 and not re.fullmatch(r"\d+f", str(s).lower()) and str(s).lower() in t:
            found.append(s)
    return list(dict.fromkeys(found))


def parse_command(text: str, proj: dict, cal: dict, default_status_day: int) -> dict:
    raw = text.strip()
    tl = raw.lower()
    if not raw:
        return {"kind": "none"}
    if re.match(r"^\s*(도움말?|help|\?|사용법|예시|명령어)", tl):
        return {"kind": "help"}
    if re.search(r"초기화|리셋|reset|원복|베이스라인으로", tl):
        return {"kind": "reset"}
    if re.search(r"되돌|실행\s*취소|undo|롤백|방금.{0,8}취소", tl):
        return {"kind": "undo"}
    if re.search(r"현황|요약|summary|status|목록", tl) and not re.search(r"\d", tl):
        return {"kind": "summary"}

    comps, sched = proj["comps"], proj["schedule"]
    s, ids, parts = raw, set(), []
    # 명시 식별자: GlobalId / #ExpressId / "부재명" / 레이어 번호
    gid_re = r"(?<![0-9A-Za-z_$])[0-9A-Za-z_$]{22}(?![0-9A-Za-z_$])"
    gids = [g for g in re.findall(gid_re, s) if g in comps.index]
    if gids:
        ids.update(gids)
        parts.append(f"GlobalId {len(gids)}개")
    s = re.sub(gid_re, " ", s)
    ex = [int(x) for x in re.findall(r"#\s*(\d+)", s)]
    if ex:
        emap = dict(zip(comps["ExpressId"], comps.index))
        hit = [emap[x] for x in ex if x in emap]
        ids.update(hit)
        parts.append(f"#ExpressId {len(hit)}개")
    s = re.sub(r"#\s*\d+", " ", s)
    for q in re.findall(r"[\"“”'‘’「」『』](.+?)[\"“”'‘’「」『』]", s):
        hit = comps.index[comps["Name"].str.contains(q, case=False, regex=False, na=False)].tolist()
        ids.update(hit)
        parts.append(f"이름 '{q}' {len(hit)}개")
    s = re.sub(r"[\"“”'‘’「」『』](.+?)[\"“”'‘’「」『』]", " ", s)
    t = s.lower()
    status_day, status_date, t = _extract_status_date(t, cal)
    if status_day is None:
        status_day = default_status_day
    layers = [int(x) for x in re.findall(r"(?:레이어|layer|단계|phase)\s*#?\s*(\d+)", t)]
    layers += [int(x) for x in re.findall(r"(\d+)\s*번?\s*째?\s*(?:레이어|단계)", t)]
    if layers:
        hit = sched.loc[sched["Layer"].isin(layers), "GlobalId"].tolist()
        ids.update(hit)
        parts.append(f"레이어 {sorted(set(layers))} {len(hit)}개")
    t = re.sub(r"(?:레이어|layer|단계|phase)\s*#?\s*\d+|\d+\s*번?\s*째?\s*(?:레이어|단계)", " ", t)

    # 공종 / 층 / 장비 필터
    cats, clss, cat_labels = set(), set(), []
    for (kind, val), words in TARGET_KEYWORDS:
        if any(re.search(_kw_regex(w), t) for w in words):
            (cats if kind == "cat" else clss).add(val)
            cat_labels.append(CAT_LABEL.get(val, {"IfcDoor": "문", "IfcWindow": "창"}.get(val, val)))
    storeys = _match_storeys(t, proj["storeys"])
    equip = [(cs, lab) for rx, cs, lab in EQUIPMENT if re.search(rx, t)]
    days = _extract_days(t, cal["wpw"])
    is_global = bool(re.search(GLOBAL_RE, t))

    # 동시 작업 한도 변경
    if re.search(NMAX_RE, t) and not days and not ids:
        m = re.search(r"(\d+)\s*(개|팀|조|건|명|tasks?)", t) or re.search(r"(\d+)", t)
        if m:
            num, after = int(m.group(1)), t[m.end():m.end() + 6]
            cur = int(proj["n_max"])
            if re.match(r"\s*(으로|로|까지)", after):
                new = num
            elif re.search(r"감소|줄|축소|빠|철수|감원|부족", t):
                new = cur - num
            elif re.search(r"증가|늘|추가|증원|보강", t):
                new = cur + num
            else:
                new = num
            return {"kind": "nmax", "value": max(1, new), "status_day": status_day, "status_date": status_date}

    # 대상 집합 결정
    targets = set(ids)
    if not targets:
        mask = pd.Series(True, index=comps.index)
        use_cats = set(cats)
        if equip and not cats and not clss:
            for cs, _ in equip:
                use_cats.update(cs)
        if use_cats or clss:
            mt = comps["Category"].isin(use_cats)
            for c in clss:
                mt |= comps["IfcType"].str.startswith(c)
            mask &= mt
        if storeys:
            mask &= comps["Storey"].isin(storeys)
        if use_cats or clss or storeys:
            targets = set(comps.index[mask])
        if storeys:
            parts.append("층 " + ", ".join(map(str, storeys)))
        if cat_labels:
            parts.append("공종 " + ", ".join(dict.fromkeys(cat_labels)))
        elif equip:
            parts.append("공종 " + ", ".join(CAT_LABEL[c] for c in sorted(use_cats, key=CAT_RANK.get)))

    if is_global and not targets:
        mode, targets, parts = "block", set(comps.index), ["전 공종"]
    elif equip:
        mode = "block"
    elif re.search(EXTEND_RE, t):
        mode = "extend"
    else:
        mode = "shift"

    if days is None:
        return {"kind": "ask", "msg": "며칠(작업일)짜리 변수인지 알려주세요. 예: `2층 벽 3일 지연`, `우천 이틀 중지`"}
    if not targets and parts:
        return {"kind": "ask", "msg": f"조건({' · '.join(parts)})에 해당하는 부재가 없습니다. "
                                      "층 이름은 TSG 탭의 노드 CSV나 공정표의 Storey 열에서 확인할 수 있어요."}
    if not targets:
        return {"kind": "ask", "msg": "대상 부재를 찾지 못했어요. 층·공종(예: `2층 벽`), `#ExpressId`, GlobalId, "
                                      "`레이어 12`, 따옴표로 감싼 부재명 중 하나를 포함해 주세요.\n\n" + HELP_TEXT}
    cause = (equip[0][1] + " 고장/불가") if equip else ("전면 작업 중지" if is_global else None)
    return {"kind": "event", "mode": mode, "targets": sorted(targets), "days": days, "status_day": status_day,
            "status_date": status_date, "desc": " · ".join(parts), "cause": cause, "raw": raw}


# ═══════════════════════════════════════════════════════════════════════════
# 7. 이벤트 적용 · 재스케줄링
# ═══════════════════════════════════════════════════════════════════════════
MODE_KO = {"shift": "착수 지연", "block": "기간 내 착수 불가", "extend": "작업기간 연장"}


def _snapshot(proj: dict) -> dict:
    return {k: copy.deepcopy(proj[k]) for k in ("schedule", "release", "extra", "n_max", "events", "last_stats")}


def apply_event(proj: dict, ev: dict, cal: dict) -> str:
    old = proj["schedule"]
    old_sum = schedule_summary(old, cal)
    sd = int(ev["status_day"])
    proj["history"].append(_snapshot(proj))
    proj["history"] = proj["history"][-20:]

    cur = old.copy()
    started = (cur["Layer_start_day"] < sd).to_numpy()
    pos = pd.Series(np.arange(len(cur)), index=cur["GlobalId"])
    N = int(ev.get("days") or 0)
    n_aff, n_ext, n_done = 0, 0, 0

    if ev["kind"] == "nmax":
        before = proj["n_max"]
        proj["n_max"] = int(ev["value"])
        head = f"동시 작업 한도 N_max **{before} → {proj['n_max']}**"
    else:
        for g in ev["targets"]:
            if g not in pos.index:
                continue
            i = int(pos[g])
            if started[i]:
                if cur.iat[i, cur.columns.get_loc("Finish_day")] <= sd:
                    n_done += 1
                    continue
                cur.iat[i, cur.columns.get_loc("Finish_day")] += N   # 진행 중 → 기간 연장
                cur.iat[i, cur.columns.get_loc("Duration_d")] += N
                n_ext += 1
            else:
                if ev["mode"] == "shift":
                    base = max(int(cur.iat[i, cur.columns.get_loc("Start_day")]), sd)
                    proj["release"][g] = max(proj["release"].get(g, 0), base + N)
                elif ev["mode"] == "block":
                    proj["release"][g] = max(proj["release"].get(g, 0), sd + N)
                else:
                    proj["extra"][g] = proj["extra"].get(g, 0) + N
                n_aff += 1
        if n_ext:  # 동결 레이어의 레이어 종료일 갱신
            lf = cur.loc[started].groupby("Layer")["Finish_day"].max()
            cur.loc[started, "Layer_finish_day"] = np.maximum(cur.loc[started, "Layer_finish_day"],
                                                              cur.loc[started, "Layer"].map(lf))
        head = f"{ev.get('cause') or MODE_KO[ev['mode']]} **{N}작업일** · 대상: {ev['desc']}"

    new, stats = reschedule(proj, sd, cur)
    proj["schedule"], proj["last_stats"] = new, stats
    new_sum = schedule_summary(new, cal)
    proj["events"].append({"시각": datetime.now().strftime("%H:%M:%S"), "입력": ev.get("raw", ""),
                           "유형": "N_max 변경" if ev["kind"] == "nmax" else MODE_KO[ev["mode"]],
                           "일수": N if ev["kind"] != "nmax" else None, "대상 수": len(ev.get("targets", [])),
                           "기준일": str(day_to_date(cal, [sd])[0].date()),
                           "공기(작업일)": f"{old_sum['total']} → {new_sum['total']}"})

    merged = old[["GlobalId", "Start_day"]].merge(new[["GlobalId", "Start_day"]], on="GlobalId", suffixes=("_o", "_n"))
    n_changed = int((merged["Start_day_o"] != merged["Start_day_n"]).sum())
    d = new_sum["total"] - old_sum["total"]
    sign = "+" if d > 0 else ""
    lines = [f"✅ 반영: {head}",
             f"- 기준일 **{day_to_date(cal, [sd])[0].date()}** (D+{sd}) · 동결 레이어 {stats['frozen_layers']}개 · "
             f"잔여 부재 재계층화 (재개 D+{stats['resume_day']})"]
    if ev["kind"] != "nmax":
        lines.append(f"- 미착수 {n_aff}개 제약 적용 · 진행 중 {n_ext}개 기간 연장" + (f" · 완료 {n_done}개 제외" if n_done else ""))
    lines += [f"- 총 공기 **{old_sum['total']} → {new_sum['total']} 작업일 ({sign}{d})**",
              f"- 준공 예정일 {old_sum['finish']} → **{new_sum['finish']}**",
              f"- 레이어 {old_sum['layers']} → {new_sum['layers']} · 일정 변동 부재 {n_changed:,}개"]
    if d == 0 and ev["kind"] != "nmax":
        lines.append("- 💡 여유(float)가 있는 경로라 준공일 영향 없이 흡수됐습니다.")
    elif d < 0:
        lines.append("- 💡 지연 부재가 빠지면서 다른 부재가 레이어를 더 촘촘히 채워 공기가 줄었습니다 "
                     "(리스트 스케줄링의 재배치 효과 — 기존 계획에 개선 여지가 있다는 신호).")
    return "\n".join(lines)


def handle_chat(prompt: str, cal: dict, status_day: int) -> str:
    proj = st.session_state.project
    if proj is None:
        return "먼저 사이드바에서 IFC 파일을 업로드해 실행하거나 **샘플 모델**을 불러와 주세요."
    ev = parse_command(prompt, proj, cal, status_day)
    k = ev["kind"]
    if k == "help":
        return HELP_TEXT
    if k == "ask":
        return ev["msg"]
    if k == "none":
        return "입력이 비어 있어요."
    if k == "reset":
        proj["history"].append(_snapshot(proj))
        proj["schedule"] = proj["baseline"].copy()
        proj["release"], proj["extra"], proj["n_max"] = {}, {}, proj["params"]["n_max"]
        proj["events"].append({"시각": datetime.now().strftime("%H:%M:%S"), "입력": prompt, "유형": "초기화"})
        s = schedule_summary(proj["schedule"], cal)
        return f"↺ 베이스라인으로 초기화했습니다. 총 공기 {s['total']} 작업일 · 준공 {s['finish']}"
    if k == "undo":
        if not proj["history"]:
            return "되돌릴 변경 이력이 없습니다."
        proj.update(proj["history"].pop())
        s = schedule_summary(proj["schedule"], cal)
        return f"↶ 직전 상태로 되돌렸습니다. 총 공기 {s['total']} 작업일 · 준공 {s['finish']}"
    if k == "summary":
        s, b = schedule_summary(proj["schedule"], cal), schedule_summary(proj["baseline"], cal)
        return (f"📊 현재 총 공기 **{s['total']}작업일** (베이스라인 {b['total']}) · 준공 {s['finish']} · "
                f"레이어 {s['layers']} · N_max {proj['n_max']} · 착수 제약 {len(proj['release'])}개 · "
                f"기간 연장 {len(proj['extra'])}개 · 적용 이벤트 {len(proj['events'])}건")
    try:
        return apply_event(proj, ev, cal)
    except Exception as e:  # 실패 시 상태 복구
        if proj["history"]:
            proj.update(proj["history"].pop())
        return f"❌ 재스케줄링 중 오류가 발생해 이전 상태를 유지합니다: `{e}`"


# ═══════════════════════════════════════════════════════════════════════════
# 8. 파이프라인 실행
# ═══════════════════════════════════════════════════════════════════════════
def new_project(source, comps, storeys, tsg, params) -> dict:
    proj = {"source": source, "comps": comps, "storeys": storeys, "G": tsg["G"], "edges": tsg["edges"],
            "conf_adj": tsg["conf_adj"], "tsg_stats": tsg["stats"], "params": params,
            "cat_days": params["cat_days"], "n_max": params["n_max"], "principle": params["principle"],
            "release": {}, "extra": {}, "events": [], "history": [], "last_stats": {}}
    sched, stats = run_dlsa(proj["G"], comps, component_durations(proj), proj["n_max"], proj["principle"],
                            conf_adj=proj["conf_adj"])
    proj["schedule"], proj["baseline"], proj["last_stats"] = sched, sched.copy(), stats
    return proj


def execute_pipeline(kind: str, params: dict, uploaded=None):
    ss = st.session_state
    with st.status("파이프라인 실행 중…", expanded=True) as status:
        log = RunLogger(status, status.progress(0.0, text="준비"))
        try:
            if kind == "ifc":
                data = uploaded.getvalue()
                key = hashlib.sha1(data).hexdigest()
                log(f"업로드: {uploaded.name} ({len(data) / 1e6:.2f} MB)")
                if key in ss.extract_cache:
                    comps, storeys, explicit = ss.extract_cache[key]
                    log("동일 파일의 추출 결과 캐시 재사용")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as tmp:
                        tmp.write(data)
                        path = tmp.name
                    try:
                        comps, storeys, explicit = extract_from_ifc(path, log)
                    finally:
                        os.remove(path)
                    ss.extract_cache = {key: (comps, storeys, explicit)}
                source = uploaded.name
            else:
                log("샘플 모델 생성: 지상 5층 RC 라멘조 (4×3 베이, 덕트·배관 교차 포함)")
                comps, storeys, explicit = build_demo_model()
                source = "샘플 모델 (5F RC)"
            log(f"부재 {len(comps):,}개 · 층 {len(storeys)}개 · 공종 분포 "
                + ", ".join(f"{CAT_LABEL[c]} {n}" for c, n in comps["Category"].value_counts().items()))

            log.progress(0.65, "TSG 구축")
            log("▶ [TSG] 위상 의미 그래프 구축")
            tsg = build_tsg(comps, explicit, params["tol"], log)

            log.progress(0.9, "DLSA 스케줄링")
            log(f"▶ [DLSA] N_max={params['n_max']} · 우선순위={params['principle']}")
            proj = new_project(source, comps, storeys, tsg, params)
            s = proj["last_stats"]
            log(f"레이어 {s['layers']}개 · 동시배치 충돌 회피 {s['conflict_deferrals']}건 · "
                f"총 공기 {int(proj['schedule']['Finish_day'].max())} 작업일")
            log.progress(1.0, "완료")
            ss.project, ss.logs = proj, log.lines
            ss.chat.append({"role": "assistant", "content":
                            f"📦 **{source}** 분석 완료 — 부재 {len(comps):,}개, 레이어 {s['layers']}개, "
                            f"총 공기 {int(proj['schedule']['Finish_day'].max())}작업일. 돌발 변수를 입력해 보세요."})
            status.update(label="✅ TSG + DLSA 완료", state="complete", expanded=False)
        except Exception as e:
            log(f"❌ 오류: {e}")
            ss.logs = log.lines
            status.update(label="❌ 파이프라인 실패", state="error", expanded=True)
            st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════
# 8.5 TSG 3D 개요 시각화 (matplotlib: 좌 3D 네트워크 + 우 엣지 관계 누적 막대)
# ═══════════════════════════════════════════════════════════════════════════
VIZ_GROUPS = [("FOUNDATION", "Foundation", "#8c564b"), ("COLUMN", "Column", "#d62728"),
              ("BEAM", "Beam", "#ff7f0e"), ("SLAB", "Slab", "#1f77b4"),
              ("WALL", "Wall", "#2ca02c"), ("OTHER", "Other", "#9e9e9e")]
SPATIAL_ORDER = [("contact", "Contact"), ("containment", "Containment"),
                 ("embedding", "Embedding"), ("non-contact", "Non-contact\n(inferred)")]
SERVICE_OF = {"support": "Support", "precedence": "Precedence", "containment": "Precedence",
              "embedding": "Precedence", "ifc_fill": "Enclosure (IFC fill)", "gravity": "Gravity support"}
SERVICE_COLORS = {"Support": "#4c72b0", "Precedence": "#dd8452",
                  "Enclosure (IFC fill)": "#55a868", "Gravity support": "#8172b3"}
VIZ_MIN_PER_RELATION = 30  # 층화 샘플링 시 관계 유형별 최소 보장 개수


def sample_edges(edges: pd.DataFrame, max_edges: int, seed: int = 0) -> pd.DataFrame:
    """관계 유형별 층화 샘플링: 비율을 유지하되 희소 관계(IFC 채움 등)도 최소 개수 보장."""
    n = len(edges)
    if n <= max_edges:
        return edges
    rng = np.random.default_rng(seed)
    groups = {r: np.asarray(ix) for r, ix in edges.groupby("Relation").indices.items()}
    floor = {r: min(len(ix), VIZ_MIN_PER_RELATION) for r, ix in groups.items()}
    alloc = {r: max(floor[r], int(round(max_edges * len(ix) / n))) for r, ix in groups.items()}
    over = sum(alloc.values()) - max_edges
    while over > 0:                                    # 초과분은 큰 그룹에서 덜어냄
        r = max(alloc, key=lambda k: alloc[k] - floor[k])
        cut = min(over, alloc[r] - floor[r])
        if cut <= 0:
            break
        alloc[r] -= cut
        over -= cut
    pick = np.concatenate([rng.choice(ix, size=min(alloc[r], len(ix)), replace=False) for r, ix in groups.items()])
    return edges.iloc[np.sort(pick)]


def build_tsg_overview_figure(comps: pd.DataFrame, edges: pd.DataFrame, max_edges: int = 3000,
                              max_nodes: int = 12000, elev: float = 22, azim: float = -60,
                              color_edges: bool = False, seed: int = 0):
    if "Spatial" not in edges.columns:  # 이전 버전 세션 결과 호환
        edges = edges.assign(Spatial=edges["Relation"].map(SPATIAL_FALLBACK).fillna("contact"))
    cx = ((comps["xmin"] + comps["xmax"]) / 2).to_numpy()
    cy = ((comps["ymin"] + comps["ymax"]) / 2).to_numpy()
    cz = ((comps["zmin"] + comps["zmax"]) / 2).to_numpy()
    known = {g[0] for g in VIZ_GROUPS}
    group = comps["Category"].where(comps["Category"].isin(known), "OTHER").to_numpy()
    n = len(comps)

    # ── 렌더링 최적화: 엣지 층화 샘플링 + (대규모 시) 노드 샘플링, 샘플 엣지 끝점은 항상 표시
    ed_s = sample_edges(edges, max_edges, seed)
    si, ti = comps.index.get_indexer(ed_s["Source"]), comps.index.get_indexer(ed_s["Target"])
    ok = (si >= 0) & (ti >= 0)
    si, ti, rel_s = si[ok], ti[ok], ed_s["Relation"].to_numpy()[ok]
    show = np.ones(n, dtype=bool)
    if n > max_nodes:
        show[:] = False
        show[np.random.default_rng(seed).choice(n, max_nodes, replace=False)] = True
        show[si] = show[ti] = True

    fig = plt.figure(figsize=(17, 8), dpi=110, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    bx = fig.add_subplot(gs[0, 1])

    # ── 좌: 3D 네트워크 — 엣지는 단일 Line3DCollection(아티스트 1개)으로 일괄 렌더링
    if len(si):
        segs = np.stack([np.column_stack([cx[si], cy[si], cz[si]]),
                         np.column_stack([cx[ti], cy[ti], cz[ti]])], axis=1)
        lw = 0.7 if len(segs) < 1500 else (0.45 if len(segs) < 6000 else 0.3)
        if color_edges:
            cols = [SERVICE_COLORS[SERVICE_OF.get(r, "Precedence")] for r in rel_s]
            lc = Line3DCollection(segs, colors=cols, linewidths=lw, alpha=0.55)
        else:
            lc = Line3DCollection(segs, colors="#6f7782", linewidths=lw, alpha=0.28)
        ax.add_collection3d(lc)
    n_show = int(show.sum())
    size = float(np.clip(9000 / max(n_show, 1), 5, 34))
    for key, label, color in VIZ_GROUPS:
        m = group == key
        if not m.any():
            continue
        mm = m & show
        ax.scatter(cx[mm], cy[mm], cz[mm], s=size, c=color, label=f"{label} ({int(m.sum()):,})",
                   depthshade=False, edgecolors="white", linewidths=0.25, alpha=0.95)

    lo = np.array([comps["xmin"].min(), comps["ymin"].min(), comps["zmin"].min()])
    hi = np.array([comps["xmax"].max(), comps["ymax"].max(), comps["zmax"].max()])
    span = np.maximum(hi - lo, 1e-3)
    pad = span * 0.04
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])
    aspect = np.maximum(span, span.max() * 0.25)   # 실제 비율 유지(너무 납작/길쭉하면 보정)
    try:
        ax.set_box_aspect(tuple(aspect), zoom=0.86)
    except TypeError:
        ax.set_box_aspect(tuple(aspect))
    for lab, setter in (("X (m)", ax.set_xlabel), ("Y (m)", ax.set_ylabel), ("Z (m)", ax.set_zlabel)):
        setter(lab, fontsize=11, labelpad=10)
    ax.tick_params(labelsize=8.5, pad=2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.97, 0.97, 0.98, 1.0))
        a.pane.set_edgecolor("#c9ccd1")
        a._axinfo["grid"].update(color="#dcdfe3", linewidth=0.6, linestyle="-")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("3D Topological Semantic Graph", fontsize=14, fontweight="bold", pad=6)
    # 범례: 좌상단 = 부재 공종, 좌하단 = 엣지
    leg_cat = ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), title="Component category",
                        fontsize=9.5, title_fontsize=10, markerscale=max(1.0, 40 / size) ** 0.5,
                        frameon=True, framealpha=0.92, edgecolor="#c9ccd1")
    if color_edges:
        present = {SERVICE_OF.get(r) for r in rel_s}
        eh = [Line2D([0], [0], color=c_, lw=2) for k_, c_ in SERVICE_COLORS.items() if k_ in present]
        el = [k_ for k_ in SERVICE_COLORS if k_ in present]
    else:
        eh, el = [Line2D([0], [0], color="#6f7782", lw=1.5, alpha=0.6)], ["dependency edge"]
    ax.add_artist(leg_cat)
    ax.legend(eh, el, loc="lower left", bbox_to_anchor=(-0.02, 0.0), title=f"Edges ({len(si):,} shown)",
              fontsize=9, title_fontsize=9.5, frameon=True, framealpha=0.92, edgecolor="#c9ccd1")

    # ── 우: 공간 관계(x축) × 서비스·지지 관계(누적) 엣지 수 — 샘플링 없이 전체 엣지 기준
    svc = edges["Relation"].map(SERVICE_OF).fillna("Precedence")
    tab = pd.crosstab(edges["Spatial"], svc)
    rows = [k for k, _ in SPATIAL_ORDER if k in tab.index] + [k for k in tab.index if k not in dict(SPATIAL_ORDER)]
    tab = tab.reindex(rows).fillna(0).astype(int)
    xlab = [dict(SPATIAL_ORDER).get(k, k) for k in rows]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for name, col in SERVICE_COLORS.items():
        if name not in tab.columns:
            continue
        vals = tab[name].to_numpy()
        bars = bx.bar(x, vals, bottom=bottom, width=0.62, color=col, edgecolor="white", linewidth=0.8, label=name)
        for b, v, b0 in zip(bars, vals, bottom):   # 큰 구간 내부 라벨
            if v and v >= 0.06 * tab.to_numpy().sum(1).max():
                bx.text(b.get_x() + b.get_width() / 2, b0 + v / 2, f"{v:,}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals
    ymax = max(bottom.max(), 1)
    for xi, tot in zip(x, bottom):
        bx.text(xi, tot + ymax * 0.015, f"{int(tot):,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    bx.set_xticks(x, xlab, fontsize=10.5)
    bx.set_ylim(0, ymax * 1.15)
    bx.set_ylabel("Edge count", fontsize=11)
    bx.set_xlabel("Spatial relation", fontsize=11)
    bx.set_title(f"Edge Relation Counts (total {len(edges):,})", fontsize=14, fontweight="bold", pad=12)
    bx.grid(axis="y", color="#e3e5e8", linewidth=0.8)
    bx.set_axisbelow(True)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    bx.legend(title="Service / support relation", loc="upper right", fontsize=9.5, title_fontsize=10,
              frameon=True, framealpha=0.92, edgecolor="#c9ccd1")
    fig.subplots_adjust(left=0.0, right=0.985, top=0.93, bottom=0.1)

    info = {"edges_total": len(edges), "edges_shown": int(len(si)), "nodes_total": n, "nodes_shown": n_show}
    return fig, info


def render_tsg_overview(proj: dict):
    if not MPL_OK:
        st.info(f"matplotlib을 불러오지 못해 3D 시각화를 생략합니다 ({MPL_ERR}). `pip install matplotlib`")
        return
    total_e = len(proj["edges"])
    with st.expander("🎛️ 3D 시각화 설정", expanded=False):
        c = st.columns(4)
        max_e = c[0].slider("표시할 최대 엣지 수", 200, 20000, 3000, 100,
                            help="초과분은 관계 유형별 층화 샘플링으로 줄여 렌더링합니다.")
        elev = c[1].slider("고도각 elev", -10, 90, 22)
        azim = c[2].slider("방위각 azim", -180, 180, -60)
        colored = c[3].toggle("엣지를 관계별 색상으로", value=False)
    key = (proj["source"], total_e, proj["tsg_stats"]["edges"], max_e, elev, azim, colored)
    cache = proj.get("_tsg_fig")
    if cache is None or cache[0] != key:     # 설정이 바뀔 때만 다시 그림(채팅 재실행 시 재사용)
        if cache is not None:
            plt.close(cache[1])
        with st.spinner("TSG 3D 네트워크 렌더링 중…"):
            fig, info = build_tsg_overview_figure(proj["comps"], proj["edges"], max_edges=max_e,
                                                  elev=elev, azim=azim, color_edges=colored)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
        proj["_tsg_fig"] = (key, fig, info, buf.getvalue())
    _, fig, info, png = proj["_tsg_fig"]
    st.pyplot(fig, clear_figure=False, dpi=170)
    sampled = info["edges_shown"] < info["edges_total"]
    st.caption(f"노드 {info['nodes_shown']:,}/{info['nodes_total']:,} · 엣지 {info['edges_shown']:,}/"
               f"{info['edges_total']:,} 표시" + (" (관계 유형별 층화 샘플링)" if sampled else "")
               + " · 우측 막대 차트는 샘플링 없이 전체 엣지 기준")
    st.download_button("⬇️ " + export_name(proj, "TSG_overview.png"), png,
                       file_name=export_name(proj, "TSG_overview.png"), mime="image/png")


# ═══════════════════════════════════════════════════════════════════════════
# 9. UI
# ═══════════════════════════════════════════════════════════════════════════
def init_state():
    ss = st.session_state
    ss.setdefault("project", None)
    ss.setdefault("logs", [])
    ss.setdefault("extract_cache", {})
    ss.setdefault("chat", [{"role": "assistant", "content":
                            "안녕하세요! IFC를 올리고 실행하면 DLSA 공정표를 만들어 드립니다. "
                            "이후 현장 돌발 변수를 말로 알려주시면 즉시 재스케줄링합니다.\n\n" + HELP_TEXT}])


def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.subheader("모델")
        up = st.file_uploader("IFC 파일 (.ifc)", type=["ifc"])
        c1, c2 = st.columns(2)
        run_btn = c1.button("▶ 실행", type="primary", disabled=(up is None or not IFC_OK))
        demo_btn = c2.button("샘플 모델")
        if not IFC_OK:
            st.caption("⚠️ ifcopenshell 미설치 — 샘플 모델만 사용 가능\n`pip install ifcopenshell`")

        st.subheader("TSG")
        tol = st.number_input("접촉 판정 허용오차 (m)", 0.0, 0.5, 0.02, 0.005, format="%.3f")

        st.subheader("DLSA")
        n_max = st.number_input("레이어당 최대 동시 작업 N_max", 1, 5000, 20, 1)
        principle = PRINCIPLES[st.radio("우선순위 원칙", list(PRINCIPLES), index=0)]
        with st.expander("공종별 기본 공기 (작업일)"):
            dur_df = pd.DataFrame({"key": list(CAT_DAYS), "공종": [CAT_LABEL[k] for k in CAT_DAYS],
                                   "작업일": list(CAT_DAYS.values())})
            ed = st.data_editor(dur_df, hide_index=True, disabled=["공종"], key="dur_editor",
                                column_config={"key": None,
                                               "작업일": st.column_config.NumberColumn(min_value=1, max_value=365, step=1)})
            cat_days = {k: int(max(1, v)) for k, v in zip(ed["key"], ed["작업일"].fillna(1))}

        st.subheader("캘린더")
        start = st.date_input("착공일", value=date.today())
        wm = WEEKMASKS[st.selectbox("근무 형태", list(WEEKMASKS), index=1)]
        hol_txt = st.text_area("휴일 (YYYY-MM-DD, 줄바꿈/쉼표 구분)", height=80,
                               placeholder="2026-12-25\n2027-01-01")
        holidays = re.findall(r"\d{4}-\d{2}-\d{2}", hol_txt or "")
        status_date = st.date_input("진도 기준일 (이전 착수 레이어 동결)", value=start)
        resched_btn = st.button("🔁 스케줄만 재계산 (TSG 재사용)", disabled=ss.project is None)

    params = {"tol": float(tol), "n_max": int(n_max), "principle": principle, "cat_days": cat_days}
    cal = make_calendar(start, wm, holidays)
    return up, run_btn, demo_btn, resched_btn, params, cal, date_to_day(cal, status_date)


def schedule_view(proj: dict, cal: dict, status_day: int) -> pd.DataFrame:
    s = proj["schedule"].merge(proj["comps"][["ExpressId", "Name", "IfcType", "Label", "Storey"]],
                               left_on="GlobalId", right_index=True, how="left")
    base = proj["baseline"].set_index("GlobalId")["Start_day"]
    s["Delta_start_d"] = (s["Start_day"] - s["GlobalId"].map(base)).fillna(0).astype(int)
    s["Start_date"] = day_to_date(cal, s["Start_day"])
    s["Finish_date"] = day_to_date(cal, s["Finish_day"] - 1)
    s["Status"] = np.select([s["Finish_day"] <= status_day, s["Start_day"] < status_day], ["완료", "진행 중"], "예정")
    cols = ["Layer", "GlobalId", "ExpressId", "Name", "IfcType", "Label", "Storey", "Duration_d",
            "Start_date", "Finish_date", "Start_day", "Finish_day", "Delta_start_d", "Status",
            "P_support", "P_impact", "P_critical"]
    return s[cols].rename(columns={"Label": "Category"})


def render_metrics(proj, cal):
    cur, base = schedule_summary(proj["schedule"], cal), schedule_summary(proj["baseline"], cal)
    ts = proj["tsg_stats"]
    c = st.columns(5)
    c[0].metric("부재 / 의존 엣지", f"{ts['nodes']:,} / {ts['edges']:,}", help=f"복잡도지수 CI = {ts['CI']}")
    d = cur["total"] - base["total"]
    c[1].metric("총 공기 (작업일)", f"{cur['total']}", delta=f"{d:+d}일" if d else None, delta_color="inverse")
    c[2].metric("준공 예정일", str(cur["finish"]),
                delta=f"{(cur['finish'] - base['finish']).days:+d} 달력일" if cur["finish"] != base["finish"] else None,
                delta_color="inverse")
    c[3].metric("레이어 수", cur["layers"], delta=(cur["layers"] - base["layers"]) or None, delta_color="inverse")
    c[4].metric("평균 동시작업 ACT", cur["act"], help=f"N_max = {proj['n_max']}")


def render_schedule_tab(proj, cal, status_day):
    view = schedule_view(proj, cal, status_day)
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1, 1])
    sto = f1.multiselect("층", proj["storeys"]["Storey"].tolist())
    cat = f2.multiselect("공종", sorted(view["Category"].dropna().unique()))
    q = f3.text_input("부재명/ID 검색")
    only = f4.toggle("변경 부재만", value=False)
    v = view
    if sto:
        v = v[v["Storey"].isin(sto)]
    if cat:
        v = v[v["Category"].isin(cat)]
    if q:
        v = v[v["Name"].str.contains(q, case=False, na=False) | v["GlobalId"].str.contains(q, na=False)
              | v["ExpressId"].astype(str).eq(q.lstrip("#"))]
    if only:
        v = v[v["Delta_start_d"] != 0]
    st.dataframe(v, hide_index=True, height=460, column_config={
        "Start_date": st.column_config.DateColumn("Start_date", format="YYYY-MM-DD"),
        "Finish_date": st.column_config.DateColumn("Finish_date", format="YYYY-MM-DD"),
        "Delta_start_d": st.column_config.NumberColumn("Δ착수(작업일)", help="베이스라인 대비 착수일 변화")})
    st.caption(f"{len(v):,} / {len(view):,}행 표시 · 날짜는 사이드바 캘린더 설정으로 즉시 재계산됩니다.")
    b1, b2 = st.columns(2)
    sched_name = export_name(proj, "DLSA_Schedule.csv")
    base_name = export_name(proj, "DLSA_Schedule_baseline.csv")
    b1.download_button(f"⬇️ {sched_name}", view.to_csv(index=False).encode("utf-8-sig"),
                       file_name=sched_name, mime="text/csv", type="primary")
    base = proj["baseline"].merge(proj["comps"][["Name", "IfcType", "Storey"]], left_on="GlobalId", right_index=True)
    base["Start_date"], base["Finish_date"] = day_to_date(cal, base["Start_day"]), day_to_date(cal, base["Finish_day"] - 1)
    b2.download_button(f"⬇️ {base_name}", base.to_csv(index=False).encode("utf-8-sig"),
                       file_name=base_name, mime="text/csv")


def render_chart_tab(proj, cal):
    if not ALT_OK:
        st.info("altair가 없어 차트를 표시할 수 없습니다. `pip install altair`")
        return
    s = proj["schedule"].merge(proj["comps"][["Label"]], left_on="GlobalId", right_index=True)
    lay = s.groupby("Layer").agg(start=("Layer_start_day", "min"), finish=("Layer_finish_day", "max"),
                                 n=("GlobalId", "size"),
                                 main=("Label", lambda x: x.value_counts().index[0])).reset_index()
    lay["Start"] = day_to_date(cal, lay["start"])
    lay["End"] = day_to_date(cal, lay["finish"] - 1) + pd.Timedelta(days=1)
    L = len(lay)
    st.markdown("**레이어 간트 차트** (색 = 레이어 주요 공종)")
    gantt = alt.Chart(lay).mark_bar(size=max(1, min(14, int(520 / max(L, 1))))).encode(
        x=alt.X("Start:T", title=None), x2="End:T",
        y=alt.Y("Layer:Q", scale=alt.Scale(reverse=True, domain=[0, L + 1]), title="Layer"),
        color=alt.Color("main:N", title="주요 공종"),
        tooltip=["Layer", alt.Tooltip("Start:T", format="%Y-%m-%d"), alt.Tooltip("End:T", format="%Y-%m-%d"),
                 alt.Tooltip("n:Q", title="부재 수"), alt.Tooltip("main:N", title="주요 공종")]).properties(height=520)
    st.altair_chart(gantt)

    def curve(df, name):
        fin = day_to_date(cal, df["Finish_day"] - 1)
        c = pd.Series(1, index=fin).groupby(level=0).sum().sort_index().cumsum()
        return pd.DataFrame({"date": c.index, "완료율(%)": c.values / len(df) * 100, "구분": name})

    sc = pd.concat([curve(proj["baseline"], "베이스라인"), curve(proj["schedule"], "현재")])
    st.markdown("**누적 완료율 S-커브** (베이스라인 vs 돌발 변수 반영)")
    st.altair_chart(alt.Chart(sc).mark_line(interpolate="step-after").encode(
        x=alt.X("date:T", title=None), y=alt.Y("완료율(%):Q"), color="구분:N",
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "구분", alt.Tooltip("완료율(%):Q", format=".1f")]
    ).properties(height=300))


def render_tsg_tab(proj):
    render_tsg_overview(proj)
    ts = proj["tsg_stats"]
    c = st.columns(4)
    c[0].metric("근접 부재쌍", f"{ts['pairs']:,}")
    c[1].metric("복잡도지수 CI", ts["CI"])
    c[2].metric("순환 제거", ts["cycles_removed"])
    c[3].metric("동시시공 금지 쌍", ts["conflicts"])
    names = proj["comps"]["Name"]
    e = proj["edges"].copy()
    e.insert(1, "SourceName", e["Source"].map(names))
    e.insert(3, "TargetName", e["Target"].map(names))
    e["Relation_ko"] = e["Relation"].map(REL_KO)
    st.dataframe(e.head(5000), hide_index=True, height=320)
    st.caption("표는 최대 5,000행 미리보기 · 전체는 CSV로 내려받으세요.")
    d1, d2 = st.columns(2)
    edges_name = export_name(proj, "TSG_edges.csv")
    nodes_name = export_name(proj, "TSG_nodes.csv")
    d1.download_button(f"⬇️ {edges_name}", e.to_csv(index=False).encode("utf-8-sig"),
                       file_name=edges_name, mime="text/csv")
    nodes = proj["comps"].reset_index()
    nodes["InDegree"] = nodes["GlobalId"].map(dict(proj["G"].in_degree()))
    nodes["OutDegree"] = nodes["GlobalId"].map(dict(proj["G"].out_degree()))
    d2.download_button(f"⬇️ {nodes_name}", nodes.to_csv(index=False).encode("utf-8-sig"),
                       file_name=nodes_name, mime="text/csv")


def render_chat_panel():
    st.markdown("#### 💬 돌발 변수 채팅")
    box = st.container(height=640)
    with box:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])


def main():
    st.set_page_config(page_title="BIM 4D · TSG×DLSA", page_icon="🏗️", layout="wide")
    init_state()
    ss = st.session_state
    up, run_btn, demo_btn, resched_btn, params, cal, status_day = sidebar()
    prompt = st.chat_input("돌발 변수 입력 (예: 2층 벽 자재 입고 3일 지연 / 우천 이틀 작업 중지 / 도움말)")

    st.title("🏗️ BIM 4D 공정 자동화 — TSG × DLSA")
    st.caption("IFC → 위상 의미 그래프(TSG) → 동적 계층 스케줄링(DLSA) → 작업일 캘린더 · 채팅 기반 실시간 재스케줄링")
    if not IFC_OK:
        st.info(f"ifcopenshell을 불러오지 못했습니다 ({IFC_ERR}). `pip install ifcopenshell` 후 재시작하세요. "
                "그 전까지는 사이드바의 **샘플 모델**로 전체 기능을 체험할 수 있습니다.")

    if run_btn and up is not None:
        ss.chat.append({"role": "user", "content": f"📎 {up.name} 분석 실행"})
        execute_pipeline("ifc", params, up)
    elif demo_btn:
        ss.chat.append({"role": "user", "content": "🧪 샘플 모델 분석 실행"})
        execute_pipeline("demo", params)
    elif resched_btn and ss.project is not None:
        p = ss.project
        p.update({"params": params, "cat_days": params["cat_days"], "n_max": params["n_max"],
                  "principle": params["principle"], "release": {}, "extra": {}, "history": [], "events": []})
        sched, stats = run_dlsa(p["G"], p["comps"], component_durations(p), p["n_max"], p["principle"],
                                conf_adj=p["conf_adj"])
        p["schedule"], p["baseline"], p["last_stats"] = sched, sched.copy(), stats
        ss.chat.append({"role": "assistant", "content": f"🔁 새 파라미터로 베이스라인을 다시 계산했습니다 "
                                                        f"(N_max {p['n_max']}, 레이어 {stats['layers']}개, "
                                                        f"총 공기 {int(sched['Finish_day'].max())}작업일)."})

    if prompt:  # 렌더링 전에 처리해 표·지표가 같은 실행에서 바로 갱신되도록
        ss.chat.append({"role": "user", "content": prompt})
        with st.spinner("DLSA 재스케줄링 중…"):
            ss.chat.append({"role": "assistant", "content": handle_chat(prompt, cal, status_day)})

    proj = ss.project
    left, right = st.columns([0.66, 0.34], gap="large")
    with left:
        if proj is None:
            st.markdown("#### 시작하기")
            st.markdown("1. 사이드바에서 `.ifc` 파일을 올리고 **▶ 실행** (또는 **샘플 모델**)\n"
                        "2. 생성된 DLSA 공정표를 확인하고 CSV로 내려받기\n"
                        "3. 오른쪽 채팅에 현장 돌발 변수를 입력 → 공기 변화 즉시 확인")
            if ss.logs:
                with st.expander("최근 실행 로그", expanded=True):
                    st.code("\n".join(ss.logs), language="text")
        else:
            st.caption(f"모델: **{proj['source']}** · 기준일 D+{status_day} "
                       f"({day_to_date(cal, [status_day])[0].date()}) · 우선순위 {proj['principle']} · N_max {proj['n_max']}")
            render_metrics(proj, cal)
            t1, t2, t3, t4, t5 = st.tabs(["📋 DLSA 공정표", "📈 공정 차트", "🕸️ TSG", "🧾 실행 로그", "⚡ 돌발 변수 이력"])
            with t1:
                render_schedule_tab(proj, cal, status_day)
            with t2:
                render_chart_tab(proj, cal)
            with t3:
                render_tsg_tab(proj)
            with t4:
                st.code("\n".join(ss.logs) or "(로그 없음)", language="text")
            with t5:
                if proj["events"]:
                    st.dataframe(pd.DataFrame(proj["events"]), hide_index=True)
                else:
                    st.caption("아직 반영된 돌발 변수가 없습니다.")
    with right:
        render_chat_panel()


if __name__ == "__main__":
    main()
