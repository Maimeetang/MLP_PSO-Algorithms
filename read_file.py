# -*- coding: utf-8 -*-
# อ่านไฟล์ .txt (คั่นด้วยแท็บหรือช่องว่าง) ที่มีคอลัมน์ 0..14
# สร้างรายการตัวอย่างในรูปแบบ:
# samples = [
#   [[x3, x6, x8, x10, x11, x12, x13, x14], [y_120, y_240]],
#   ...
# ]
# โดย y_120 = ค่า attribute 5 ที่ i+120, y_240 = ค่า attribute 5 ที่ i+240
# แถวที่ไม่มีคู่ (เกินขอบ) จะถูกลบออก
# หมายเหตุ: ถ้าต้องการตัดค่าที่เป็น -200 ออก ให้ตั้ง drop_neg200=True

from typing import List, Tuple

def _to_float(token: str) -> float:
    # รองรับทศนิยม comma
    return float(token.replace(",", ".").strip())

def build_samples_from_txt(
    path: str,
    feature_indices: List[int] = [3, 6, 8, 10, 11, 12, 13, 14],
    target_index: int = 5,
    horizons: Tuple[int, ...] = (120,),  # รองรับ horizon เดียวหรือหลายตัว
    drop_neg200: bool = False
) -> List[List[List[float]]]:

    rows: List[List[str]] = []

    # อ่านไฟล์
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            rows.append(parts)

    n = len(rows)
    if n == 0:
        return []

    # เตรียมฟีเจอร์และเป้าหมาย
    features_all: List[List[float]] = []
    target_all: List[float] = []

    for parts in rows:
        try:
            feats = [_to_float(parts[i]) for i in feature_indices]
            tgt = _to_float(parts[target_index])
        except Exception:
            feats, tgt = None, None

        features_all.append(feats) # type: ignore
        target_all.append(tgt) # type: ignore

    # --- ส่วนสร้าง samples ---
    samples: List[List[List[float]]] = []
    max_h = max(horizons)
    last_i = n - 1 - max_h

    for i in range(max(0, last_i + 1)):
        feats = features_all[i]

        # สร้าง output ตามจำนวน horizon ที่กำหนด
        ys = [target_all[i + h] for h in horizons]

        # ข้าม None
        if feats is None or any(y is None for y in ys):
            continue

        if drop_neg200:
            if any(v == -200 for v in feats) or any(y == -200 for y in ys):
                continue

        samples.append([feats, ys])

    return samples



# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    path = "AirQualityUCI.txt"  # เปลี่ยนเป็นพาธไฟล์ .txt ของคุณ
    samples = build_samples_from_txt(
        path=path,
        feature_indices=[3, 6, 8, 10, 11, 12, 13, 14],
        target_index=5,
        horizons=(120, 240),
        drop_neg200=False  # ตั้ง True เพื่อตัดค่าที่เป็น -200 ออก
    )

    print("จำนวนตัวอย่างที่ได้:", len(samples))
    if samples:
        print("ตัวอย่างแถวแรก:")
        print(samples[8040])  # รูปแบบ: [[x3,x6,...,x14], [y_120, y_240]]