# -*- coding: utf-8 -*-
"""Build the detailed Thai status PDF: advisor comments, problems found, fixed, outstanding, and plan.

Every figure is transcribed from a committed file under E:/thesis/reports or models/. Nothing
is estimated. Rebuild after any measurement changes:
    python docs/reports/make_detailed_status_report.py
"""
import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

OUT = pathlib.Path(r"E:\thesis\reports\detailed_status_2026-11-01.pdf")

pdfmetrics.registerFont(TTFont("TH", r"C:\Windows\Fonts\leelawad.ttf"))
pdfmetrics.registerFont(TTFont("TH-Bold", r"C:\Windows\Fonts\leelawdb.ttf"))
pdfmetrics.registerFontFamily("TH", normal="TH", bold="TH-Bold")

INK = colors.HexColor("#17191c")
MUTED = colors.HexColor("#5b6471")
ACCENT = colors.HexColor("#1d4ed8")
GOOD = colors.HexColor("#15803d")
WARN = colors.HexColor("#b45309")
BAD = colors.HexColor("#b91c1c")
RULE = colors.HexColor("#d6dae2")
BAND = colors.HexColor("#f4f6f9")

ss = getSampleStyleSheet()


def st(name, size, leading, **kw):
    o = dict(fontName="TH", textColor=INK, spaceBefore=0, spaceAfter=0)
    o.update(kw)
    return ParagraphStyle(name, parent=ss["BodyText"], fontSize=size, leading=leading,
                          alignment=TA_LEFT, **o)


S = {
    "title": st("title", 22, 30, fontName="TH-Bold"),
    "sub": st("sub", 11, 16.5, textColor=MUTED),
    "h1": st("h1", 15.5, 21, fontName="TH-Bold", textColor=ACCENT, spaceBefore=15, spaceAfter=6),
    "h2": st("h2", 12.4, 17.5, fontName="TH-Bold", spaceBefore=10, spaceAfter=4),
    "h3": st("h3", 11, 16, fontName="TH-Bold", textColor=colors.HexColor("#334155"),
             spaceBefore=8, spaceAfter=3),
    "body": st("body", 10.4, 16, spaceAfter=6),
    "bullet": st("bullet", 10.4, 16, leftIndent=13, bulletIndent=3, spaceAfter=3),
    "quote": st("quote", 11.2, 17.5, fontName="TH-Bold", textColor=colors.HexColor("#7c2d12")),
    "note": st("note", 9.5, 14.2, textColor=MUTED, spaceAfter=5),
    "cell": st("cell", 8.9, 12.6),
    "cellb": st("cellb", 8.9, 12.6, fontName="TH-Bold"),
    "cellh": st("cellh", 8.9, 12.6, fontName="TH-Bold", textColor=colors.white),
}

flow = []
A = flow.append
W = 162 * mm


def h1(t): A(Paragraph(t, S["h1"]))
def h2(t): A(Paragraph(t, S["h2"]))
def h3(t): A(Paragraph(t, S["h3"]))
def p(t): A(Paragraph(t, S["body"]))
def note(t): A(Paragraph(t, S["note"]))
def gap(h=5): A(Spacer(1, h))


def bullets(items, style="bullet"):
    for it in items:
        A(Paragraph(it, S[style], bulletText="•"))
    gap(4)


def box(head, body_lines, border=ACCENT, bg=colors.HexColor("#eef2ff")):
    inner = [[Paragraph(head, S["quote"])]]
    for ln in body_lines:
        inner.append([Paragraph(ln, S["body"])])
    t = Table(inner, colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, border),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    A(t)
    gap(7)


def table(header, rows, widths, highlight=None, cmap=None, fs=None):
    cellh, cell, cellb = S["cellh"], S["cell"], S["cellb"]
    if fs:
        cellh = ParagraphStyle("hh", parent=S["cellh"], fontSize=fs, leading=fs + 3.4)
        cell = ParagraphStyle("cc", parent=S["cell"], fontSize=fs, leading=fs + 3.4)
        cellb = ParagraphStyle("cb", parent=S["cellb"], fontSize=fs, leading=fs + 3.4)
    data = [[Paragraph(c, cellh) for c in header]]
    for i, r in enumerate(rows):
        base = cellb if (highlight and i in highlight) else cell
        row = []
        for c in r:
            cs = base
            if cmap and i in cmap:
                cs = ParagraphStyle(f"m{i}", parent=base, textColor=cmap[i])
            row.append(Paragraph(c, cs))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    stl = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            stl.append(("BACKGROUND", (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(stl))
    A(t)
    gap(7)


# ════════════════════════════════════════════════════════════ COVER
A(Paragraph("รายงานสถานะโดยละเอียด", S["title"]))
A(Paragraph("Hybrid IDS สำหรับ Campus SOC<br/>"
            "คอมเมนต์อาจารย์ · ปัญหาที่พบ · สิ่งที่แก้แล้ว · สิ่งที่ยังไม่แก้ และแนวทาง<br/>"
            "ฉบับวันที่ 1 พฤศจิกายน 2569", S["sub"]))
gap(12)

h1("สารบัญ")
table(
    ["ส่วน", "เนื้อหา", "หน้า"],
    [
        ["1", "คอมเมนต์อาจารย์ และสิ่งที่คอมเมนต์นั้นเปิดเผย", "2"],
        ["2", "ปัญหาที่พบทั้งหมด — จัดกลุ่มตามความรุนแรง", "3–4"],
        ["3", "สิ่งที่แก้แล้ว พร้อมหลักฐาน", "5–6"],
        ["4", "สิ่งที่ยังไม่แก้ และแนวทางแก้", "7–9"],
        ["5", "สิ่งที่แก้ไม่ได้ด้วยข้อมูลชุดนี้ (ข้อจำกัดเชิงโครงสร้าง)", "10"],
        ["6", "ลำดับงานที่แนะนำ", "11"],
    ],
    [16 * mm, 118 * mm, 20 * mm],
)

box("หลักการเขียนรายงานนี้", [
    "ทุกตัวเลขคัดลอกจากไฟล์ผลลัพธ์ที่ commit แล้วใน <font face='TH-Bold'>reports/</font> หรือ "
    "<font face='TH-Bold'>models/</font> ไม่มีการประมาณ",
    "รายการ “ยังไม่แก้” แยกเป็นสองแบบที่ต่างกันโดยสิ้นเชิง: "
    "<b>(ก) แก้ได้แต่ยังไม่ได้ทำ</b> และ <b>(ข) แก้ไม่ได้ด้วยข้อมูล/ทรัพยากรที่มี</b> — "
    "การรวมสองแบบนี้เข้าด้วยกันจะทำให้อ่านสถานะผิด",
    "ข้อที่เป็น <b>trade-off ที่เลือกแล้ว</b> ไม่ใช่บั๊ก ก็แยกไว้อีกกลุ่ม เพราะต้องรายงานในเล่ม "
    "ไม่ใช่รอแก้",
])

# ════════════════════════════════════════════════════════════ 1
A(PageBreak())
h1("1. คอมเมนต์อาจารย์ และสิ่งที่คอมเมนต์นั้นเปิดเผย")

box("คอมเมนต์ที่ 1", [
    "“ไม่สามารถรวม unsupervised และ supervised dataset ได้ ดังนั้นไปพิจารณา dataset ที่นำมาใช้ "
    "และ วิธีการวัดประสิทธิภาพใหม่ค่ะ”",
], border=colors.HexColor("#c2410c"), bg=colors.HexColor("#fff7ed"))

p("<b>คอมเมนต์นี้ถูกต้อง และกลไกมีอยู่จริง</b> ไม่ใช่ความเข้าใจผิด:")
table(
    ["โมเดล", "อ่านชุดข้อมูล", "จำนวน feature"],
    [
        ["Random Forest (supervised)", "MachineLearningCSV", "80"],
        ["Isolation Forest + Autoencoder (unsupervised)", "GeneratedLabelledFlows (behavioral)", "92"],
    ],
    [62 * mm, 62 * mm, 30 * mm],
)
p("สองชุดนี้เป็นการสกัดด้วย CICFlowMeter <b>สองครั้งแยกกัน</b> จาก capture เดียวกัน ไม่มีคีย์เชื่อม "
  "ต้องใช้ row-hash join ซึ่งจับคู่แถว BENIGN ได้ <font color='#b91c1c'><b>76.4%</b></font> "
  "ทำให้ประเมิน ensemble ได้เพียง <font color='#b91c1c'><b>80.36%</b></font> ของ test split "
  "แล้วเอาไปเทียบกับ RF ที่เห็นข้อมูล 100%")

box("คอมเมนต์นี้เปิดเผยปัญหาที่ใหญ่กว่าที่ถาม", [
    "พอแก้การ join แล้ววัดใหม่อย่างยุติธรรม <b>กฎรวมผลที่ใช้อยู่เดิมกลายเป็นแพ้ Random Forest "
    "ตัวเดียว</b> จึงต้องเปลี่ยนสถาปัตยกรรมหลักของชั้น ML — ไม่ใช่แค่เปลี่ยนไฟล์ข้อมูล",
    "สาเหตุที่แท้จริงคือ margin 9.3pp ของกฎเดิม <b>มาจากการที่ RF ถูกจำกัด feature ไว้</b> "
    "ไม่ใช่จากตัวกฎ การเทียบเดิมไม่เป็นธรรมกับ RF มาตลอด (พิสูจน์ด้วย ablation ในส่วน 3)",
], border=BAD, bg=colors.HexColor("#fef2f2"))

box("คอมเมนต์ที่ 2", [
    "“ทำไมเลือกใช้ ML สามตัวนี้ เพราะมันมีหลายโมเดลมาก”",
], border=colors.HexColor("#c2410c"), bg=colors.HexColor("#fff7ed"))
p("คำตอบเดิม<b>อ่อน เพราะไม่เคยวัดคู่เทียบเลย</b> — การเลือก RF อาศัยความนิยมในงานวิจัย ไม่ใช่หลักฐาน "
  "จากการทดลองของเราเอง")

# ════════════════════════════════════════════════════════════ 2
A(PageBreak())
h1("2. ปัญหาที่พบทั้งหมด")

h2("กลุ่ม A — ปัญหาเชิงวิธีวิจัย (ทำให้ตัวเลขที่รายงานไม่ถูกต้อง)")
table(
    ["#", "ปัญหา", "ผลกระทบ", "สถานะ"],
    [
        ["A1", "สอง population ต้อง join · coverage 80.36%",
         "เทียบ ensemble กับ RF ไม่ยุติธรรม", "แก้แล้ว"],
        ["A2", "LOAO normalize คะแนน test fold ด้วย min/max ของ test fold เอง",
         "test distribution รั่วเข้าคะแนน · ทุก candidate สูงเกินจริง 1–4pp", "แก้แล้ว"],
        ["A3", "normalization constants fit บน val ไม่ใช่ train (ae_max 0.811 เทียบค่าจริง 0.1225)",
         "ถ้า deploy จะทำให้ ae_norm เล็กลง 6.6 เท่า · threshold ไม่มีทางถึง · "
         "detector เงียบโดยไม่มี error", "แก้แล้ว"],
        ["A4", "SIEM baseline วัดบน test split คนละชุดกับ ensemble",
         "ตารางหลักของ 4-way ablation เทียบของต่างชุดกัน", "แก้แล้ว"],
        ["A5", "ข้อสรุป Option A ว่า behavioral features ไม่ช่วย RF",
         "เป็นการให้เหตุผลตามหลักการ ไม่เคยวัด · วัดแล้วพบว่าผิด", "แก้แล้ว"],
        ["A6", "in-dataset metric อิ่มตัว (RF val F1 0.9997)",
         "แยกโมเดลไม่ออก ต้องใช้ zero-day LOAO เท่านั้น", "ข้อจำกัด"],
    ],
    [10 * mm, 52 * mm, 62 * mm, 22 * mm],
    cmap={5: WARN}, fs=8.4,
)

h2("กลุ่ม B — ปัญหาเชิงวิศวกรรม (โค้ดพังเงียบ หรือจะพังเมื่อ deploy)")
table(
    ["#", "ปัญหา", "ผลกระทบ", "สถานะ"],
    [
        ["B1", "`model_used` whitelist ใน alert_dispatcher ไม่ตรงกับ fusion_layer",
         "alert ทุกใบที่ branch นั้นตัดสินถูกติดป้าย “unknown” เงียบๆ", "แก้แล้ว"],
        ["B2", "shap_explain เขียนไว้สำหรับกฎเดิม · ส่ง np.zeros แทนคะแนน IF",
         "ถูกต้องตอนใช้ OR (IF ไม่มีผล) แต่ cascade ใช้ IF จริง → attribution ผิดแบบดูไม่ออก",
         "แก้แล้ว"],
        ["B3", "synthetic overlay สร้างใน 92-feature space ขาด Source Port / Protocol",
         "โมเดลที่ deploy score overlay ไม่ได้เลย · M1b วัดไม่ได้", "แก้แล้ว"],
        ["B4", "`mttd_seconds()` มีอยู่แต่ไม่มีสคริปต์ไหนเรียก",
         "M1b อยู่ในรายการ metric แต่เป็นค่าว่าง", "แก้แล้ว"],
        ["B5", "M1c ไม่มีโมเดลให้รัน (RF 80f score ชุด 76 คอลัมน์ไม่ได้)",
         "M1c วัดไม่ได้", "แก้แล้ว"],
        ["B6", "autoencoder_unified_config.json พา benign_row_count_reduction_pct 23.57 มาผิด",
         "เลขนี้กลับทิศกับความจริง (unified มี BENIGN มากกว่า) · เสี่ยงหลุดเข้าเล่ม", "แก้แล้ว"],
        ["B7", "random_forest_unified.pkl 108 MB เกินลิมิต GitHub",
         "push พัง · เคยต้องล้าง history เพราะ blob 483 MB มาแล้ว", "แก้แล้ว"],
        ["B8", "few-shot ใน prompt ถูกคัดลอกเข้า output 30/100 alert",
         "คำอธิบายอ้าง IP/hostname ที่ไม่มีใน alert จริง", "แก้แล้ว"],
        ["B9", "eval_llm_faithfulness.py ยังโหลดโมเดลและ population ที่ถูก supersede",
         "ตัวเลข M2 ที่มีอยู่อธิบาย pipeline ที่ไม่ได้ deploy แล้ว", "ยังไม่แก้"],
        ["B10", "eda_sanity_check.py ยังอ่าน UNSW-NB15 รูปแบบเก่า 4 ไฟล์",
         "รายงาน EDA ไม่ตรงกับสิ่งที่เข้า pipeline จริง", "ยังไม่แก้"],
    ],
    [10 * mm, 52 * mm, 62 * mm, 22 * mm],
    cmap={8: BAD, 9: WARN}, fs=8.4,
)

h2("กลุ่ม C — อุบัติเหตุปฏิบัติการ")
table(
    ["#", "เหตุการณ์", "ผล / บทเรียน", "สถานะ"],
    [
        ["C1", "`rm -rf` บน git worktree ที่มี Windows junction ชี้ไป data/",
         "ลบทะลุ junction ทำลาย data/ จริง · ต้อง rebuild ทั้ง pipeline · "
         "กู้คืนได้เพราะทุกขั้นตอน pin random_state=42 (drift 0.00pp)", "บันทึกแล้ว"],
    ],
    [10 * mm, 52 * mm, 62 * mm, 22 * mm], fs=8.4,
)
note("บทเรียน C1 บันทึกไว้ใน CLAUDE.md แล้ว: ถ้าต้องแชร์ data/ เข้า worktree ให้ลบ junction ด้วย "
     "`rmdir` ห้ามใช้ `rm -rf` หรือดีกว่านั้นคือให้สคริปต์อ่าน absolute path ไม่ต้องสร้าง junction")

# ════════════════════════════════════════════════════════════ 3
A(PageBreak())
h1("3. สิ่งที่แก้แล้ว พร้อมหลักฐาน")

h2("3.1 แก้ที่ชั้นข้อมูล — เลิก join (A1)")
table(
    ["", "ก่อน", "หลัง"],
    [
        ["จำนวน population", "2 ชุด สกัดแยกกัน", "1 ชุด"],
        ["การ join", "row-hash join จำเป็น", "ไม่มีเลย"],
        ["Coverage test split", "80.36%", "100.00%"],
        ["Feature set", "RF 80 / IF+AE 92", "94 ทุกโมเดล"],
        ["Train / Val / Test", "ไม่ตรงกัน", "1,780,362 / 504,192 / 504,193"],
    ],
    [46 * mm, 55 * mm, 58 * mm], highlight={2, 3},
)
note("สคริปต์: src/features/build_unified_population.py → src/modeling/train_unified_ensemble.py · "
     "คอลัมน์ระบุตัวตน 5 ตัวถูกตัดจาก input (Source IP จะทำให้โมเดลจำได้ว่า host ไหนโจมตี = leakage)")

h2("3.2 เปลี่ยนกฎรวมผล และพิสูจน์สาเหตุ (A5)")
table(
    ["กฎ (nested LOAO)", "F1", "Recall", "FPR"],
    [
        ["confidence_weighted ← ใช้จริง", "0.9782", "0.9635", "0.0007"],
        ["OR: RF หรือ AE (2 thresholds)", "0.9745", "0.9608", "0.0011"],
        ["RF ตัวเดียว", "0.9737", "0.9593", "0.0011"],
        ["OR: max(RF, AE) ← กฎเดิม", "0.9724", "0.9559", "0.0010"],
        ["AE ตัวเดียว", "0.8736", "0.8176", "0.0055"],
        ["IF ตัวเดียว", "0.4371", "0.4502", "0.0619"],
    ],
    [74 * mm, 28 * mm, 28 * mm, 28 * mm], highlight={0}, cmap={3: BAD},
)
h3("ablation ที่พิสูจน์สาเหตุ — ตัดเฉพาะ 12 behavioral columns ออก")
table(
    ["RF ตัวเดียว ในแต่ละเงื่อนไข", "F1", "Recall"],
    [
        ["unified 94 features (มี behavioral)", "0.9737", "0.9593"],
        ["unified 82 features (ตัด behavioral)", "0.8312", "0.7942"],
        ["population เก่าที่ join · 80 features", "0.8327", "0.7705"],
    ],
    [78 * mm, 40 * mm, 40 * mm],
)
p("ตัด behavioral ออกแล้ว RF <b>ตกกลับไปทับ population เก่าแทบสนิท</b> (0.8312 เทียบ 0.8327) — "
  "สาเหตุคือ feature ไม่ใช่ population และไม่ใช่ตัวกฎ")
note("ผลไม่สม่ำเสมอทุกหมวด: ที่ argmax behavioral columns ทำให้ RF แย่ลงบน DoS Slowhttptest "
     "(0.055 เทียบ 0.275) และ DoS slowloris (0.224 เทียบ 0.455) จึงต้องรายงานต่อหมวด")

h2("3.3 ตอบคอมเมนต์ที่ 2 ด้วยการวัด")
table(
    ["โมเดล", "เวลาเทรน", "Inference 504k แถว", "F1", "AUC"],
    [
        ["Random Forest (sklearn, CPU)", "231.5 s", "5.45 s", "0.9997", "1.0000"],
        ["XGBoost (GPU)", "38.7 s", "0.94 s", "0.9907", "0.9961"],
    ],
    [54 * mm, 26 * mm, 36 * mm, 23 * mm, 23 * mm], highlight={0},
)
p("XGBoost ไม่ชนะด้านการตรวจจับ แต่ <b>เทรนเร็วกว่า 6 เท่า inference เร็วกว่า 5.8 เท่า</b> "
  "ที่ RF ช้าเพราะ scikit-learn ไม่มี GPU backend เลย — เป็นข้อจำกัดของไลบรารี ต้องรายงานคู่กัน")

h2("3.4 วิธีวัดใหม่ — 4-way ablation บน test split เดียวกัน (A4)")
table(
    ["Arm", "Precision", "Recall", "F1", "FPR"],
    [
        ["Rule-based SIEM", "0.8297", "0.8980", "0.8625", "0.0375"],
        ["Hybrid (confidence_weighted)", "0.9952", "0.9981", "0.9966", "0.00098"],
        ["RF ตัวเดียว", "0.9995", "0.9995", "0.9995", "0.00009"],
    ],
    [56 * mm, 26 * mm, 26 * mm, 26 * mm, 26 * mm],
)
note("n = 504,193 แถว ทุก arm · ค่ารวมของ SIEM หลอกตา: 8 จาก 14 หมวดได้ 0.000 เป๊ะ · "
     "rule R1 (port scan) ยิงโดนแถว attack ศูนย์แถว แต่สร้าง false positive 4,240 ครั้ง")

h2("3.5 metric ที่วัดครบในรอบนี้")
table(
    ["Metric", "ผล", "ข้อควรระวังที่ต้องเขียนคู่กัน"],
    [
        ["M1a in-dataset", "Hybrid F1 0.9966 · RF 0.9995",
         "อิ่มตัว แยกโมเดลไม่ออก"],
        ["zero-day LOAO", "Hybrid F1 0.9782 · RF 0.9737",
         "ส่วนต่างแค่ 0.45pp"],
        ["M1b MTTD", "c2_beacon 193 s · data_exfil 452 s",
         "ค่าหลักเป็น 0.0000 s ซึ่ง degenerate ต้องใช้ค่าที่ตัด flow แรกออก"],
        ["M1c cross-dataset", "CICIDS→UNSW F1 0.0006 · UNSW→CICIDS 0.1139",
         "พังทั้งสองทาง · AUC 0.3220 คือจัดอันดับกลับทิศ"],
        ["SHAP faithfulness", "RF 1.96e-10 · fallback 4.66e-09",
         "latency 0.258 / 0.184 s ต่อ alert ไม่เหมาะ bulk"],
    ],
    [34 * mm, 58 * mm, 70 * mm], fs=8.4,
)

# ════════════════════════════════════════════════════════════ 4
A(PageBreak())
h1("4. สิ่งที่ยังไม่แก้ และแนวทางแก้")

p("แยกตามว่า<b>ทำเองได้ทันที</b> หรือ<b>ต้องรอทรัพยากรภายนอก</b> — เพราะสองอย่างนี้วางแผนต่างกัน")

h2("4.1 ทำเองได้ทันที — ควรทำก่อนส่งเล่ม")

box("B9 · eval_llm_faithfulness.py อยู่บน pipeline ที่ถูก supersede  [ความสำคัญ: สูง]", [
    "<b>ปัญหา:</b> สคริปต์โหลด <font face='TH-Bold'>random_forest.pkl</font> (80f), "
    "<font face='TH-Bold'>isolation_forest_behavioral_v2.pkl</font>, "
    "<font face='TH-Bold'>autoencoder_behavioral_v2.pt</font> และอ่านไฟล์ feature สองชุดพร้อม "
    "row-hash join — ซึ่งเป็นสิ่งที่ unified population ยกเลิกไปแล้ว ค่า `model_used` "
    "ที่มันบันทึกคือ `autoencoder_v2` ซึ่งไม่มีอยู่อีกแล้ว",
    "<b>ผลคือ</b> ตัวเลข M2 ที่มีอยู่ (entity preservation 28/100 ฯลฯ) "
    "อธิบาย pipeline ที่ไม่ได้ deploy — ยังอ้างในเล่มไม่ได้",
    "<b>แนวทาง:</b> ย้ายไปอ่าน <font face='TH-Bold'>cicids2017_unified_test.parquet</font> + "
    "โมเดล <font face='TH-Bold'>*_unified</font> ทั้งสามตัว ตัด row-hash join ออก "
    "(ไม่ต้อง join แล้ว) และแก้การ recover IP/port/timestamp ให้ดึงจาก identifier columns "
    "ที่ unified population มีอยู่ตรงๆ แล้ววัด M2 ใหม่บน 100 alert",
    "<b>ประเมินเวลา:</b> แก้สคริปต์ ~1 ชม. · รัน 100 alert ผ่าน Typhoon2 8B ~30–45 นาที",
], border=BAD, bg=colors.HexColor("#fef2f2"))

box("คุณภาพผลผลิต LLM ยังไม่ผ่านการตรวจ  [ความสำคัญ: สูง]", [
    "<b>สถานะ:</b> โค้ดเสร็จและผ่าน 6 offline contract checks แต่ "
    "<font face='TH-Bold'>quality_verified</font> ยังยืนยันบน alert จริงไม่เกิน 2 เคส "
    "และ <b>Ollama ไม่ได้รันอยู่</b> ตอนตรวจ (port 11434 ปิด) จึงวัดอะไรไม่ได้เลยในขณะนี้",
    "<b>ปัญหาที่วัดได้ก่อนหน้านี้และยังไม่ยืนยันว่าหายแล้ว:</b> entity omission "
    "(ละ IP/port ที่ต้องมี) 72 จาก 100 เคสที่ fail · PortScan remediation แนะนำ WAF block "
    "ใน 5/7 เคส ซึ่งไม่ตรงกับลักษณะการโจมตี",
    "<b>แนวทาง:</b> (1) start Ollama + ยืนยันว่า pull "
    "<font face='TH-Bold'>scb10x/llama3.1-typhoon2-8b-instruct</font> แล้ว "
    "(2) แก้ B9 ก่อน (3) รัน M2 ใหม่ (4) ให้คนอ่านภาษาไทยประเมินเชิงคุณภาพ "
    "ตัวอย่างละ 20–30 เคสต่อหมวดสำคัญ — ข้อ (4) เป็นงานที่สคริปต์แทนไม่ได้",
], border=BAD, bg=colors.HexColor("#fef2f2"))

box("B10 · eda_sanity_check.py อ่าน UNSW รูปแบบเก่า  [ความสำคัญ: ต่ำ–กลาง]", [
    "<b>ปัญหา:</b> ยังอ่าน <font face='TH-Bold'>UNSW-NB15_1..4.csv</font> (รูปแบบ 4 ไฟล์ headerless) "
    "ขณะที่ pipeline จริงใช้ CIC-UNSW-NB15 (<font face='TH-Bold'>Data.csv + Label.csv</font>) "
    "มาตั้งแต่ migration — รายงาน EDA จึงอธิบายข้อมูลที่ไม่ได้เข้า pipeline",
    "<b>แนวทาง:</b> ชี้ `load_unsw` ไปที่ source เดียวกับ preprocess_pipeline "
    "(import มาใช้ซ้ำ ไม่เขียนใหม่ เพื่อไม่ให้ drift) แล้ว regenerate eda_report.txt",
    "<b>ประเมินเวลา:</b> ~30 นาที · <b>ความเสี่ยงถ้าไม่แก้:</b> ถ้าเล่มอ้างสถิติ EDA "
    "ของ UNSW จะเป็นสถิติของไฟล์ที่ไม่ได้ใช้",
], border=WARN, bg=colors.HexColor("#fffbeb"))

box("เอกสาร CLAUDE.md ย่อหน้า “Current implementation state” อ่านยาก  [ความสำคัญ: ต่ำ]", [
    "<b>ปัญหา:</b> ย่อหน้านี้เล่าประวัติ confidence_weighted → rf_ae_or → confidence_weighted "
    "ซ้อนกันสามชั้น ถูกต้องทุกคำแต่คนอ่านครั้งแรกจะสับสน และ notebooks 01–08 "
    "ทั้งหมดอธิบาย population ที่ถูก supersede",
    "<b>แนวทาง:</b> ตัดย่อหน้าให้เหลือสถานะปัจจุบัน แล้วย้ายประวัติไปไว้หัวข้อ "
    "“superseded history” หัวข้อเดียว",
], border=MUTED, bg=BAND)

# ════════════════════════════════════════════════════════════ 4.2
A(PageBreak())
h2("4.2 ต้องรอทรัพยากรภายนอก — วางแผนล่วงหน้าได้แต่ทำเองไม่ได้")

table(
    ["งาน", "ติดอะไร", "แนวทาง / สิ่งที่ต้องเตรียม"],
    [
        ["A/B analyst triage-time study<br/>(≥15 analysts × 20 alerts, blinded)",
         "ต้องมีผู้เข้าร่วมจริง และต้องผ่านคณะกรรมการจริยธรรมการวิจัยในมนุษย์",
         "เตรียมชุด alert 2 กลุ่ม (hybrid / baseline) ให้พร้อมและ blind ไว้ล่วงหน้า · "
         "ร่างเอกสารขอจริยธรรม + แบบยินยอม · กำหนดตัวชี้วัดเวลาและความถูกต้องก่อนเก็บข้อมูล "
         "(ไม่ใช่หลังเห็นผล)"],
        ["ส่ง LINE / Gmail จริง",
         "sandbox ไม่มี outbound internet · Gmail OAuth2 เว้นว่างไว้โดยเจตนา",
         "ทดสอบจากเครื่องที่ออกเน็ตได้ · ขอ OAuth2 client + refresh token แล้วใส่ใน "
         "notification/.env (ไม่ commit) · เส้นทาง LINE มี token จริงแล้ว เหลือแค่เน็ต"],
        ["Kafka broker จริง + OpenCTI จริง กับ enrich.py",
         "ยังไม่เคยต่อของจริงเลย · `kafka-python` ยังไม่ติดตั้ง",
         "ติดตั้ง kafka-python · รัน stack ตามคำสั่งใน CLAUDE.md · "
         "<b>คาดว่า GraphQL query shapes จะต้องแก้</b> เพราะเขียนจากความรู้ 5.12.x "
         "โดยไม่เคยยิงจริง (query ผิดจะกลายเป็น “ไม่ match” ไม่ใช่ match ผิด จึงไม่อันตราย "
         "แต่ต้องตรวจ)"],
        ["ตรวจคุณภาพ LLM เชิงภาษา",
         "ต้องมีคนอ่านภาษาไทยตัดสินว่าคำอธิบายใช้งานได้จริงไหม",
         "ทำหลังแก้ B9 · สุ่ม 20–30 เคสต่อหมวดสำคัญ · ให้คะแนนตามเกณฑ์ที่กำหนดไว้ก่อนอ่าน "
         "(ความถูกต้องของ entity · ความเหมาะสมของ remediation · ความชัดเจน)"],
    ],
    [40 * mm, 48 * mm, 74 * mm], fs=8.3,
)

h2("4.3 สิ่งที่แก้แบบ “เลี่ยง” ไม่ใช่แก้ที่ต้นเหตุ")
box("MTTD ค่าหลัก degenerate — ตอนนี้แก้ด้วยการตัด flow แรกออก", [
    "<b>สภาพจริง:</b> flow แรกของทั้ง 228 campaign มี behavioral vector เป็นศูนย์ทั้ง 12 คอลัมน์ "
    "(ถูกต้องทางกายภาพ) และกฎให้คะแนนเฉลี่ย 8.94 เทียบ threshold 0.03 → ตรวจจับได้ทุกอัน → "
    "MTTD = 0.0000 s",
    "<b>วิธีรับมือตอนนี้:</b> รายงานค่าที่ตัด flow แรกของแต่ละ campaign ออก พร้อมอธิบายเหตุผล "
    "— ได้ตัวเลขที่แยกแยะได้ (c2_beacon 193 s · data_exfil 452 s)",
    "<b>แก้ที่ต้นเหตุจะทำอย่างไร:</b> ให้ generator ตั้งค่า window features ของ flow แรก "
    "จากการแจกแจงของ BENIGN จริง แทนที่จะเป็นศูนย์ — แต่ต้องตัดสินก่อนว่า "
    "“ผู้โจมตีจากภายนอกที่เพิ่งปรากฏตัว” ควรมีค่า window เป็นศูนย์จริงหรือไม่ "
    "ซึ่งเป็นคำถามเชิงแบบจำลอง ไม่ใช่เชิงโค้ด",
], border=WARN, bg=colors.HexColor("#fffbeb"))

box("overlay ให้ค่า behavioral อ่อนกว่าข้อมูลจริงมาก", [
    "<b>วัดได้:</b> `conn_count_1m` ของ overlay สูงสุด <b>0.014</b> เทียบข้อมูลจริง <b>1.0</b> · "
    "`uniq_dport_count_1m` 0.32 เทียบ 1.0 — port scan สังเคราะห์อ่อนกว่าของจริงมาก "
    "บน feature ที่ Option A เพิ่มเข้ามาโดยตรง",
    "<b>ผลคือ</b> ห้ามอ่านอัตราการตรวจจับบน overlay เป็นหลักฐานความสามารถ (เขียนกำกับไว้แล้ว)",
    "<b>แนวทางแก้:</b> ปรับ PROFILES ให้ความเข้มของ window features ใกล้การแจกแจงจริง "
    "โดย fit จาก attack rows จริงของแต่ละหมวด แทนการตั้งค่าจากกลไกการโจมตีด้วยมือ",
], border=WARN, bg=colors.HexColor("#fffbeb"))

# ════════════════════════════════════════════════════════════ 5
A(PageBreak())
h1("5. สิ่งที่แก้ไม่ได้ด้วยข้อมูลชุดนี้")

p("กลุ่มนี้<b>ไม่ใช่งานค้าง</b> แต่เป็นข้อจำกัดเชิงโครงสร้างที่ต้องเขียนในเล่มเป็นข้อจำกัดของงาน")

table(
    ["ข้อจำกัด", "ตัวเลข / สภาพจริง", "ทำไมแก้ไม่ได้"],
    [
        ["Bot ตรวจไม่ได้เลย", "nested LOAO recall 0.000 · ทุกสมาชิกเกือบศูนย์",
         "ไม่มีกฎรวมใดกู้คืนได้เพราะไม่มีสมาชิกไหนตรวจเจอ · ต้องการ feature "
         "หรือข้อมูลชนิดใหม่ ไม่ใช่กฎใหม่"],
        ["สองหมวดไม่มี redundancy",
         "FTP-Patator และ Web Attack - BF ขึ้นกับ RF ตัวเดียว · PortScan ขึ้นกับ AE ตัวเดียว",
         "เป็นผลจากความสามารถของโมเดลแต่ละชนิด ไม่ใช่การตั้งค่า"],
        ["M1c พังทั้งสองทาง",
         "CICIDS→UNSW AUC 0.3220 (กลับทิศ) · UNSW→CICIDS AUC 0.5034 (สุ่ม)",
         "สอง testbed ต่างกัน เครื่องมือโจมตีต่างกัน เก็บต่างปีกัน · "
         "เป็นผลการวิจัยที่ต้องรายงาน ไม่ใช่การทดลองที่ล้มเหลว"],
        ["M1c วัดได้แค่ RF",
         "IF/AE เข้าร่วมไม่ได้",
         "behavioral features ต้องใช้ Source IP / Timestamp ซึ่ง CIC-UNSW-NB15 "
         "ไม่มีในไฟล์ ML-ready"],
        ["in-dataset metric อิ่มตัว", "RF val F1 0.9997",
         "เป็นคุณสมบัติของ CICIDS-2017 เอง · แก้โดยใช้ zero-day LOAO เป็นตัวตัดสินแทน"],
        ["PA-5220 log จริงไม่มี", "อยู่นอกขอบเขตถาวร (ตัดสินใจ 25 ต.ค. 2569)",
         "ไม่เคยได้รับข้อมูล · CICIDS-2017 เป็นชุดเทรนชุดเดียว"],
        ["IF ถูกตัดจากการตัดสินหลัก",
         "IF ดีที่สุดบน DoS Slowhttptest (0.839 เทียบ AE 0.241, RF 0.055) แต่ FPR 0.09–0.15",
         "เป็น trade-off ที่วัดแล้วและเลือกแล้ว ไม่ใช่ข้อบกพร่อง · ต้องรายงานว่าเสียอะไรไป"],
        ["hybrid แพ้ RF ตัวเดียวใน dataset",
         "F1 0.9966 เทียบ 0.9995 · FPR แย่กว่า 10 เท่า",
         "เป็น trade-off ที่เลือกเพราะชนะ zero-day · ส่วนต่างแคบพอที่ผู้อ่าน "
         "จะเลือก RF ตัวเดียวอย่างมีเหตุผลได้ ต้องเขียนตรงๆ"],
    ],
    [36 * mm, 56 * mm, 70 * mm], fs=8.3,
)

# ════════════════════════════════════════════════════════════ 6
A(PageBreak())
h1("6. ลำดับงานที่แนะนำ")

table(
    ["ลำดับ", "งาน", "เหตุผลที่จัดลำดับนี้", "เวลา"],
    [
        ["1", "แก้ eval_llm_faithfulness.py ไป unified (B9)",
         "บล็อกทุกอย่างที่เกี่ยวกับ LLM · ตัวเลข M2 ที่มีอยู่ยังอ้างไม่ได้", "~1 ชม."],
        ["2", "start Ollama + รัน M2 ใหม่ 100 alert",
         "ได้ตัวเลข M2 ที่ตรงกับ pipeline ที่ deploy จริง", "~45 นาที"],
        ["3", "ตรวจคุณภาพ LLM เชิงภาษาโดยคนอ่าน",
         "layer เดียวที่โค้ดเสร็จแต่ไม่มีหลักฐานคุณภาพเลย · ทำได้โดยไม่ต้องรอใครภายนอก",
         "ครึ่งวัน"],
        ["4", "แก้ eda_sanity_check.py (B10)",
         "ถูกเพราะเล่มอาจอ้างสถิติ EDA · ความเสี่ยงต่ำแต่ราคาถูก", "~30 นาที"],
        ["5", "เตรียมชุด alert + เอกสารจริยธรรมสำหรับ A/B study",
         "เป็นเส้นทางวิกฤต (critical path) ที่สุด เพราะรอคณะกรรมการ · เริ่มเตรียมได้เลย",
         "1–2 วัน + รอ"],
        ["6", "ทดสอบ LINE / Gmail จากเครื่องที่ออกเน็ตได้",
         "ปิดช่องว่าง “dry-run only” ของชั้น output", "~2 ชม."],
        ["7", "ต่อ Kafka + OpenCTI จริงกับ enrich.py",
         "คาดว่าต้องแก้ GraphQL query shapes · ไม่บล็อกตัวเลขในเล่ม", "ครึ่งวัน"],
        ["8", "จัดระเบียบ CLAUDE.md + ปรับ overlay profiles",
         "งานคุณภาพเอกสาร/แบบจำลอง ไม่บล็อกการส่งเล่ม", "~2 ชม."],
    ],
    [12 * mm, 46 * mm, 76 * mm, 22 * mm], highlight={0, 1, 2}, fs=8.3,
)

box("ข้อสังเกตเรื่องลำดับ", [
    "งานที่ <b>รอคนอื่น</b> (A/B study ข้อ 5) ควรเริ่ม<b>เตรียมพร้อมกัน</b>กับข้อ 1–3 ไม่ใช่ทำตามลำดับ "
    "เพราะเวลารอคณะกรรมการจริยธรรมเป็นส่วนที่ควบคุมไม่ได้และยาวที่สุด",
    "ข้อ 1–3 เป็นชุดเดียวกัน: ทำข้อ 2 ก่อนข้อ 1 จะได้ตัวเลขที่ต้องทิ้งอยู่ดี",
])

gap(8)
h2("สรุปสถานะเป็นตัวเลข")
table(
    ["หมวด", "จำนวน"],
    [
        ["ปัญหาที่พบและแก้แล้ว", "14 ข้อ (A1–A5, B1–B8, C1)"],
        ["ยังไม่แก้ · ทำเองได้", "4 ข้อ (B9, B10, คุณภาพ LLM, เอกสาร)"],
        ["ยังไม่แก้ · รอทรัพยากรภายนอก", "4 ข้อ"],
        ["แก้แบบเลี่ยง ยังไม่ถึงต้นเหตุ", "2 ข้อ (MTTD flow แรก, overlay profiles)"],
        ["ข้อจำกัดเชิงโครงสร้าง (รายงานในเล่ม)", "8 ข้อ"],
    ],
    [84 * mm, 62 * mm],
)

note("ไฟล์หลักฐาน: reports/unified_val_metrics.csv · fusion_layer_loao_unified_or_comparison.csv · "
     "rf_behavioral_ablation_loao_report.txt · baseline_model_comparison.csv · "
     "siem_baseline_report.txt · cross_dataset_m1c_metrics.csv · "
     "mttd_metrics_excluding_first_flow.csv · shap_explainability_report.txt · "
     "models/fusion_config.json · models/llm_layer_config.json")


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("TH", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(24 * mm, 12 * mm, "Hybrid IDS — รายงานสถานะโดยละเอียด · 1 พ.ย. 2569")
    canvas.drawRightString(186 * mm, 12 * mm, f"หน้า {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(24 * mm, 16 * mm, 186 * mm, 16 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4,
                      leftMargin=24 * mm, rightMargin=24 * mm,
                      topMargin=18 * mm, bottomMargin=22 * mm,
                      title="รายงานสถานะโดยละเอียด — Hybrid IDS",
                      author="Teerapong P.")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
doc.build(flow)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
