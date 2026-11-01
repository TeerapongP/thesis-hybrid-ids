# -*- coding: utf-8 -*-
"""Build the Thai PDF report: what the advisor asked, what was changed, and what it measured.

Every number here is read from or transcribed off a committed report file under E:/thesis/reports.
Nothing is estimated.
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
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

OUT = pathlib.Path(r"E:\thesis\reports\advisor_response_2026-11-01.pdf")

pdfmetrics.registerFont(TTFont("Leelawadee", r"C:\Windows\Fonts\leelawad.ttf"))
pdfmetrics.registerFont(TTFont("Leelawadee-Bold", r"C:\Windows\Fonts\leelawdb.ttf"))
pdfmetrics.registerFontFamily("Leelawadee", normal="Leelawadee", bold="Leelawadee-Bold")

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6471")
ACCENT = colors.HexColor("#1d4ed8")
GOOD = colors.HexColor("#15803d")
WARN = colors.HexColor("#b45309")
BAD = colors.HexColor("#b91c1c")
RULE = colors.HexColor("#d6dae2")
BAND = colors.HexColor("#f4f6f9")

ss = getSampleStyleSheet()


def st(name, size, leading, **kw):
    opts = dict(fontName="Leelawadee", textColor=INK, spaceBefore=0, spaceAfter=0)
    opts.update(kw)
    return ParagraphStyle(name, parent=ss["BodyText"], fontSize=size, leading=leading,
                          alignment=TA_LEFT, **opts)


S = {
    "title": st("title", 23, 31, fontName="Leelawadee-Bold"),
    "sub": st("sub", 11.5, 17, textColor=MUTED),
    "h1": st("h1", 16, 22, fontName="Leelawadee-Bold", textColor=ACCENT,
             spaceBefore=16, spaceAfter=7),
    "h2": st("h2", 12.8, 18, fontName="Leelawadee-Bold", spaceBefore=11, spaceAfter=5),
    "body": st("body", 10.6, 16.4, spaceAfter=7),
    "bullet": st("bullet", 10.6, 16.4, leftIndent=13, bulletIndent=3, spaceAfter=4),
    "quote": st("quote", 11.4, 18, fontName="Leelawadee-Bold", textColor=colors.HexColor("#7c2d12")),
    "note": st("note", 9.7, 14.6, textColor=MUTED, spaceAfter=6),
    "cell": st("cell", 9.2, 13),
    "cellb": st("cellb", 9.2, 13, fontName="Leelawadee-Bold"),
    "cellh": st("cellh", 9.2, 13, fontName="Leelawadee-Bold", textColor=colors.white),
}

flow = []
A = flow.append


def h1(t): A(Paragraph(t, S["h1"]))
def h2(t): A(Paragraph(t, S["h2"]))
def p(t): A(Paragraph(t, S["body"]))
def note(t): A(Paragraph(t, S["note"]))
def gap(h=5): A(Spacer(1, h))


def bullets(items):
    for it in items:
        A(Paragraph(it, S["bullet"], bulletText="•"))
    gap(4)


def box(lines, border=ACCENT, bg=colors.HexColor("#eef2ff"), style="body"):
    inner = []
    for i, ln in enumerate(lines):
        inner.append([Paragraph(ln, S[style] if i else S["quote"])])
    t = Table(inner, colWidths=[162 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 3, border),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    A(t)
    gap(8)


def table(header, rows, widths, highlight=None, colors_map=None):
    """highlight: set of row indices (0-based over `rows`) to bold. colors_map: {row_idx: color}."""
    data = [[Paragraph(c, S["cellh"]) for c in header]]
    for i, r in enumerate(rows):
        stl = "cellb" if (highlight and i in highlight) else "cell"
        row = []
        for c in r:
            cs = S[stl]
            if colors_map and i in colors_map:
                cs = ParagraphStyle(f"c{i}", parent=S[stl], textColor=colors_map[i])
            row.append(Paragraph(c, cs))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
        ("GRID", (0, 0), (-1, -1), 0.35, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), BAND))
    if highlight:
        for i in highlight:
            style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#dcfce7")))
    t.setStyle(TableStyle(style))
    A(t)
    gap(8)


# ══════════════════════════════════════════════════════════════════════
A(Paragraph("การตอบคอมเมนต์อาจารย์ที่ปรึกษา", S["title"]))
A(Paragraph("Hybrid IDS สำหรับ Campus SOC — สรุปสิ่งที่แก้ไข วิธีแก้ และผลที่วัดได้<br/>"
            "ฉบับวันที่ 1 พฤศจิกายน 2569", S["sub"]))
gap(13)

box([
    "คอมเมนต์ที่ 1 (อาจารย์)",
    "“ไม่สามารถรวม unsupervised และ supervised dataset ได้ ดังนั้นไปพิจารณา dataset "
    "ที่นำมาใช้ และ วิธีการวัดประสิทธิภาพใหม่ค่ะ”",
], border=colors.HexColor("#c2410c"), bg=colors.HexColor("#fff7ed"))

box([
    "คอมเมนต์ที่ 2 (อาจารย์)",
    "“ทำไมเลือกใช้ ML สามตัวนี้ เพราะมันมีหลายโมเดลมาก”",
], border=colors.HexColor("#c2410c"), bg=colors.HexColor("#fff7ed"))

h1("สรุปสั้นที่สุด")
p("<b>คอมเมนต์ที่ 1 ถูกต้อง และกลไกที่อาจารย์ชี้มีอยู่จริง</b> ไม่ได้แก้ด้วยการอธิบายว่าของเดิมพอใช้ได้ "
  "แต่แก้ที่ชั้นข้อมูล — <b>เลิกการ join สองชุดข้อมูลทิ้งทั้งหมด</b> ให้ทั้งสามโมเดลอ่านไฟล์เดียวกัน "
  "ชุด feature เดียวกัน และ split เดียวกัน")
p("การแก้นี้ทำให้เกิดผลที่<b>ไม่ได้คาดไว้และเปลี่ยนสถาปัตยกรรมหลัก</b>: เมื่อวัดใหม่อย่างยุติธรรม "
  "กฎรวมผลที่ใช้อยู่เดิมกลายเป็นแพ้ Random Forest ตัวเดียว จึงต้องเปลี่ยนกฎ — "
  "รายละเอียดอยู่ในหัวข้อ 2 และ 3")
p("<b>คอมเมนต์ที่ 2 ตอบด้วยการวัด ไม่ใช่การอ้างอิง</b> เพิ่ม XGBoost เป็นคู่เทียบบนข้อมูลชุดเดียวกัน "
  "ผลคือ XGBoost ไม่ชนะ จึงมีหลักฐานว่าทำไมเลือก RF — แต่ XGBoost เร็วกว่า 6 เท่า ซึ่งต้องรายงานด้วย")

# ══════════════════════════════════════════════════════════════════════
A(PageBreak())
h1("1. คอมเมนต์ที่ 1 — ปัญหาจริงคืออะไร")

p("ก่อนแก้ ระบบอ่านข้อมูลจาก <b>สองชุดที่ถูกสกัดแยกกัน</b> ด้วย CICFlowMeter จาก capture เดียวกัน:")
bullets([
    "Random Forest (supervised) อ่านชุด <font face='Leelawadee-Bold'>MachineLearningCSV</font> → 80 features",
    "Isolation Forest + Autoencoder (unsupervised) อ่านชุด "
    "<font face='Leelawadee-Bold'>GeneratedLabelledFlows</font> → 92 features (มี behavioral features ที่ต้องใช้ IP/Timestamp)",
])
p("สองชุดนี้ไม่มีคีย์เชื่อมกัน ต้องใช้ <b>row-hash join</b> ซึ่งจับคู่แถว BENIGN ได้เพียง "
  "<font color='#b91c1c'><b>76.4%</b></font> ผลคือ ensemble ประเมินได้แค่ "
  "<font color='#b91c1c'><b>80.36%</b></font> ของ test split แล้วนำไปเทียบกับ RF ที่เห็นข้อมูล 100%")
box([
    "เหตุผลที่คอมเมนต์อาจารย์ถูก",
    "การเทียบ ensemble (80.36% ของข้อมูล) กับ RF (100% ของข้อมูล) ไม่ใช่การเทียบที่ยุติธรรมตั้งแต่ต้น "
    "และ 405,142 แถวที่ใช้ประเมินเป็น artifact ของการ join ไฟล์ ไม่ใช่สิ่งที่ระบบจริงจะเห็น",
], border=colors.HexColor("#b91c1c"), bg=colors.HexColor("#fef2f2"))

h1("2. วิธีแก้ — เลิก join ไม่ใช่เถียงว่า join ดีพอ")

p("ไฟล์ <font face='Leelawadee-Bold'>cicids2017_behavioral_labeled.parquet</font> (2,520,961 แถว) "
  "มีทุกคอลัมน์ที่ทั้งสามโมเดลต้องใช้อยู่แล้ว จึง <b>split 60/20/20 ตรงจากไฟล์นี้</b> "
  "(<font face='Leelawadee-Bold'>src/features/build_unified_population.py</font>)")

table(
    ["", "ก่อนแก้", "หลังแก้"],
    [
        ["จำนวนชุดข้อมูล", "2 ชุด สกัดแยกกัน", "1 ชุด"],
        ["การ join", "row-hash join จำเป็น", "ไม่มีการ join เลย"],
        ["Coverage ของ test split", "80.36%", "100.00%"],
        ["Feature set", "RF 80 / IF+AE 92 (คนละชุด)", "94 ทุกโมเดล (ชุดเดียวกัน)"],
        ["Train / Val / Test", "ไม่ตรงกันระหว่างโมเดล", "1,780,362 / 504,192 / 504,193"],
    ],
    [52 * mm, 55 * mm, 55 * mm],
    highlight={2, 3},
)
note("หมายเหตุ: คอลัมน์ระบุตัวตน 5 ตัว (Flow ID, Source IP, Destination IP, Timestamp, day_group) "
     "ถูกตัดออกจาก input ของโมเดล — โดยเฉพาะ Source IP ถ้าปล่อยไว้โมเดลจะจำได้ว่า host ไหนโจมตีใน "
     "capture นี้ ซึ่งเป็น data leakage ที่ปลอมตัวมาเป็น feature")

p("<b>ข้อจำกัดที่การแก้นี้ไม่ได้ปลดล็อก:</b> CIC-UNSW-NB15 ไม่มีคอลัมน์ Source IP / Timestamp "
  "จึงคำนวณ behavioral features 12 ตัวให้มันไม่ได้ ดังนั้น IF และ AE ยังเข้าร่วมการทดสอบข้าม dataset "
  "(M1c) ไม่ได้ — ดูหัวข้อ 6")

# ══════════════════════════════════════════════════════════════════════
A(PageBreak())
h1("3. ผลที่ไม่ได้คาดไว้ — กฎรวมผลต้องเปลี่ยน")

p("เมื่อวัดใหม่บนข้อมูลชุดเดียวกันด้วย nested LOAO (เลือก threshold จาก 9 หมวดที่เหลือ "
  "แล้วไปใช้กับหมวดที่กันไว้) <b>กฎเดิมที่ใช้อยู่แพ้ Random Forest ตัวเดียว</b>")

table(
    ["กฎรวมผล (candidate)", "F1 (zero-day)", "Recall", "FPR"],
    [
        ["confidence_weighted ← ใช้จริงตอนนี้", "0.9782", "0.9635", "0.0007"],
        ["OR: RF หรือ AE (2 thresholds)", "0.9745", "0.9608", "0.0011"],
        ["RF ตัวเดียว", "0.9737", "0.9593", "0.0011"],
        ["OR: max(RF, AE) ← กฎเดิม", "0.9724", "0.9559", "0.0010"],
        ["AND: RF และ AE", "0.9504", "0.9062", "0.0001"],
        ["AE ตัวเดียว", "0.8736", "0.8176", "0.0055"],
        ["IF ตัวเดียว", "0.4371", "0.4502", "0.0619"],
    ],
    [72 * mm, 32 * mm, 29 * mm, 29 * mm],
    highlight={0},
    colors_map={3: BAD},
)

h2("หาสาเหตุด้วย ablation ไม่ใช่เดา")
p("เดิม กฎ OR ชนะ RF ตัวเดียวถึง 9.3pp — แต่ตอนนั้น RF ถูกจำกัดให้ใช้แค่ 80 features "
  "ไม่มี behavioral features เลย จึงทดสอบโดย<b>ตัดเฉพาะ 12 behavioral columns ออก</b> "
  "แล้วรัน LOAO ชุดเดิมซ้ำ (<font face='Leelawadee-Bold'>rf_behavioral_ablation_loao.py</font>)")

table(
    ["RF ตัวเดียว ในแต่ละเงื่อนไข", "F1 (zero-day)", "Recall"],
    [
        ["unified, 94 features (มี behavioral)", "0.9737", "0.9593"],
        ["unified, 82 features (ตัด behavioral ออก)", "0.8312", "0.7942"],
        ["population เก่าที่ join, 80 features", "0.8327", "0.7705"],
    ],
    [78 * mm, 42 * mm, 42 * mm],
)
box([
    "ข้อสรุปของ ablation",
    "ตัด behavioral columns ออกแล้ว RF <b>ตกกลับไปทับ population เก่าแทบสนิท</b> (0.8312 เทียบ 0.8327) "
    "แปลว่า margin 9.3pp ของกฎ OR เดิม <b>มาจากการที่ RF ถูกมัดมือไว้</b> ไม่ใช่จากตัวกฎ — "
    "การเทียบเดิมไม่เป็นธรรมกับ RF มาตลอด",
], border=GOOD, bg=colors.HexColor("#f0fdf4"))

p("นอกจากนี้ยัง<b>ขัดกับข้อสรุปเดิมที่บันทึกไว้</b>ว่า behavioral features ไม่ช่วย RF "
  "เพราะเป็นข้อจำกัดของ supervised generalization ไม่ใช่เรื่องการแทน feature — "
  "ข้อสรุปนั้นมาจากการให้เหตุผลตามหลักการ <b>ไม่เคยวัด</b> เพราะ RF ไม่เคยได้รับ 12 คอลัมน์นั้นเลย")
note("แต่ผลไม่สม่ำเสมอทุกหมวด: ที่ argmax behavioral columns ทำให้ RF แย่ลงบน DoS Slowhttptest "
     "(0.055 เมื่อมี เทียบ 0.275 เมื่อไม่มี) และ DoS slowloris (0.224 เทียบ 0.455) "
     "จึงต้องรายงานต่อหมวด ไม่ใช่สรุปรวมว่า “ช่วย”")

h2("กฎที่ใช้จริงตอนนี้")
box([
    "confidence_weighted — RF นำ มี unsupervised รองรับ",
    "<font face='Courier'>rf_confidence = |P(attack) − 0.5| × 2</font><br/>"
    "ถ้า <font face='Courier'>rf_confidence ≥ 0.95</font> → ใช้คะแนนของ RF<br/>"
    "ถ้าไม่ถึง → ใช้ <font face='Courier'>(if_norm + ae_norm) / 2</font><br/>"
    "แจ้งเตือนเมื่อคะแนนรวม &gt; 0.03",
])
p("<b>ทั้งสามโมเดลมีส่วนร่วมจริง</b> ซึ่งทำให้กรอบ “three-model ensemble” ยังใช้ได้ "
  "แต่ Isolation Forest ตัดสินเฉพาะแถวที่ RF ไม่มั่นใจ เพราะ FPR ของมันอยู่ที่ 0.09–0.15 "
  "สูงกว่า RF ประมาณพันเท่า ใช้เป็นตัวจุดชนวนอิสระไม่ได้")
note("เหตุผลที่เคยปัดกฎนี้ตกไปใช้ไม่ได้แล้ว: ตอนนั้น tuner เลือก confidence_threshold = 1.00 ใน 9/10 fold "
     "ซึ่งเท่ากับปิดการทำงานของตัวเอง รอบนี้เลือก 0.95 ใน 9/10 fold แปลว่าทางรองรับทำงานจริง — "
     "ตรวจข้อนี้ก่อนตัดสินใจรับกฎ ไม่ใช่สมมติ")

# ══════════════════════════════════════════════════════════════════════
A(PageBreak())
h1("4. คอมเมนต์ที่ 2 — ทำไมเลือก ML สามตัวนี้")

p("คำตอบเดิมอ่อน เพราะ<b>ไม่เคยวัดคู่เทียบเลย</b> การเลือก Random Forest อาศัยความนิยม ไม่ใช่หลักฐาน "
  "จึงเพิ่ม XGBoost ซึ่งเป็นคู่แข่งที่ชัดที่สุด วัดบน population / feature / split เดียวกันทุกอย่าง "
  "(<font face='Leelawadee-Bold'>baseline_model_comparison.py</font>)")

table(
    ["โมเดล", "เวลาเทรน", "Inference (504k แถว)", "F1", "AUC"],
    [
        ["Random Forest (sklearn, CPU)", "231.5 s", "5.45 s", "0.9997", "1.0000"],
        ["XGBoost (GPU)", "38.7 s", "0.94 s", "0.9907", "0.9961"],
    ],
    [56 * mm, 26 * mm, 36 * mm, 22 * mm, 22 * mm],
    highlight={0},
)

p("<b>XGBoost ไม่ชนะด้านการตรวจจับ</b> และ per-category recall แพ้ RF เกือบทุกหมวด "
  "(Bot 0.931 เทียบ 0.972 · PortScan 0.986 เทียบ 1.000 · SSH-Patator 0.983 เทียบ 1.000) "
  "เหตุผลที่เลือก RF จึงกลายเป็น “วัดแล้วดีกว่า” แทน “เป็นที่นิยม”")

box([
    "สิ่งที่ต้องพูดคู่กันเสมอ ไม่ใช่โชว์แค่คอลัมน์ F1",
    "XGBoost <b>เทรนเร็วกว่า 6 เท่า และ inference เร็วกว่า 5.8 เท่า</b> ถ้าเกณฑ์คือ latency ในระบบจริง "
    "XGBoost ชนะชัดเจน ที่ RF ช้าเพราะ scikit-learn <b>ไม่มี GPU backend เลย</b> "
    "เป็นข้อจำกัดของไลบรารี ไม่ใช่การตั้งค่าที่ไม่ดี",
], border=WARN, bg=colors.HexColor("#fffbeb"))

p("และสิ่งที่การวัดนี้<b>ตอบได้จริงมีเพียงข้อเดียว</b>: ไม่มีทางเลือกที่ยังไม่ได้ทดสอบที่ดีกว่าอย่างชัดเจน "
  "<b>ไม่ได้แปลว่า RF ดีที่สุด</b> — เพราะ in-dataset บน CICIDS-2017 อิ่มตัวแล้ว ต่างกันระดับพันส่วนของ F1 "
  "ไม่พิสูจน์อะไร ตัวที่แยกโมเดลออกจริงคือ zero-day LOAO")
note("ไม่เพิ่ม One-Class SVM โดยเจตนา: ฝั่ง unsupervised มีคำตอบอยู่แล้ว เพราะ IF และ AE "
     "มี per-category recall correlation 0.925–0.977 (ซ้ำซ้อนกันสูง) และ LOAO ablation "
     "ก็ตัด IF ออกจากการตัดสินใจหลักไปแล้ว")

# ══════════════════════════════════════════════════════════════════════
A(PageBreak())
h1("5. วิธีวัดประสิทธิภาพใหม่ — 4-way ablation บน test split เดียวกัน")

p("คอมเมนต์อาจารย์สั่งให้พิจารณา “วิธีการวัดประสิทธิภาพใหม่” ด้วย จึงพบและแก้<b>ข้อผิดพลาดในตารางหลัก</b>: "
  "rule-based SIEM baseline ยังอ่าน population เก่า ขณะที่ ensemble ย้ายไป unified แล้ว "
  "ตัวเลขสองตัวที่เอามาเทียบกัน<b>มาจาก test set คนละชุด</b>")

table(
    ["Arm ของ ablation", "Precision", "Recall", "F1", "FPR"],
    [
        ["Rule-based SIEM", "0.8297", "0.8980", "0.8625", "0.0375"],
        ["Hybrid (confidence_weighted)", "0.9952", "0.9981", "0.9966", "0.00098"],
        ["RF ตัวเดียว", "0.9995", "0.9995", "0.9995", "0.00009"],
    ],
    [60 * mm, 26 * mm, 26 * mm, 26 * mm, 24 * mm],
)
note("ทุกตัวเลขวัดบน test split เดียวกัน n = 504,193 แถว")

h2("ค่ารวมของ SIEM หลอกตา ต้องรายงานต่อหมวด")
p("F1 0.8625 มาจาก DDoS 0.988 · DoS Hulk 0.983 · PortScan 0.933 เป็นหลัก ขณะที่ "
  "<b>8 จาก 14 หมวดได้ 0.000 เป๊ะ</b> (Bot, FTP-/SSH-Patator, Web Attack ทั้งสามแบบ, Heartbleed, "
  "Infiltration) และอีกสองหมวดเกือบศูนย์ (DoS slowloris 0.006 · DoS Slowhttptest 0.007)")
bullets([
    "rule R1 (port scan) ยิงโดนแถว attack <b>ศูนย์แถว</b> แต่สร้าง false positive 4,240 ครั้ง — "
    "recall ของ PortScan มาจาก rule R3 (volume) ไม่ใช่ R1",
    "rule R2 (brute force) target recall 0.000 แต่ไปจับ DoS Hulk และ DDoS แทน",
    "rule R5 (beaconing) ไม่ยิงอะไรเลย",
])

h2("จุดที่ hybrid แพ้ และต้องเขียนในเล่ม")
box([
    "ต้องรายงานสองทาง ไม่ใช่ทางเดียว",
    "<b>zero-day LOAO:</b> hybrid 0.9782 ชนะ RF ตัวเดียว 0.9737 (+0.45pp) ที่ FPR ดีกว่า "
    "(0.0007 เทียบ 0.0011)<br/>"
    "<b>in-dataset:</b> hybrid 0.9966 <b>แพ้</b> RF ตัวเดียว 0.9995 และ FPR แย่กว่า 10 เท่า "
    "(0.00098 เทียบ 0.00009)<br/><br/>"
    "ข้อสรุปที่ตรงไปตรงมาคือ hybrid แลก precision ใน dataset กับความสามารถ generalize ไปหา zero-day "
    "และบน dataset นี้ส่วนต่างแคบพอที่ผู้อ่านจะเลือก RF ตัวเดียวอย่างมีเหตุผลได้",
], border=WARN, bg=colors.HexColor("#fffbeb"))

# ══════════════════════════════════════════════════════════════════════
A(PageBreak())
h1("6. M1c — ทดสอบข้าม dataset แล้วพังทั้งสองทาง")

p("วัดด้วย <font face='Leelawadee-Bold'>eval_cross_dataset.py</font> รายงาน "
  "<b>สามแถวต่อทิศ ไม่ใช่สองแถว</b> เพราะคะแนนข้าม dataset เดี่ยวๆ แยกไม่ออกว่า "
  "“generalize ไม่ได้” หรือ “ชุด feature ร่วมอ่อนกว่า”")

table(
    ["ทิศ", "โมเดล", "วัดบน", "F1", "AUC", "FPR"],
    [
        ["CICIDS→UNSW", "full 80f", "CICIDS test", "0.9958", "0.9998", "0.0009"],
        ["CICIDS→UNSW", "shared 76f", "CICIDS test", "0.9958", "0.9999", "0.0011"],
        ["CICIDS→UNSW", "shared 76f", "UNSW test", "0.0006", "0.3220", "0.0000"],
        ["UNSW→CICIDS", "full 76f", "UNSW test", "0.9551", "0.9933", "0.0274"],
        ["UNSW→CICIDS", "shared 76f", "UNSW test", "0.9551", "0.9933", "0.0274"],
        ["UNSW→CICIDS", "shared 76f", "CICIDS test", "0.1139", "0.5034", "0.2103"],
    ],
    [28 * mm, 24 * mm, 28 * mm, 22 * mm, 22 * mm, 22 * mm],
    colors_map={2: BAD, 5: BAD},
)

p("<b>การพังเป็น generalization ทั้งหมด ไม่ใช่ผลจากการตัด feature</b> — ตัดลงเหลือ shared set "
  "เสีย F1 เพียง −0.0001 ทิศหนึ่ง และ 0.0000 อีกทิศ ถ้าไม่มีแถวกลางนี้จะเถียงไม่ได้เลย")
bullets([
    "<b>CICIDS→UNSW: AUC 0.3220 ต่ำกว่า 0.5</b> คือ<b>การจัดอันดับกลับทิศ</b> — โมเดลให้คะแนนแถว attack "
    "ว่า benign มากกว่าแถว benign จริงอย่างเป็นระบบ รุนแรงกว่าแค่ generalize ไม่ได้ "
    "(precision 1.0000 ในแถวนั้น<b>ห้ามนำไปอ้าง</b> มันเป็น artifact ของการยิงแทบไม่เคยยิง)",
    "<b>UNSW→CICIDS: AUC 0.5034</b> คือสุ่ม พร้อม FPR 0.2103 ยิงโดน benign หนึ่งในห้า",
])
box([
    "สิ่งที่ M1c สร้างให้วิทยานิพนธ์",
    "ขอบเขตของข้อกล่าวอ้างแบบ dataset เดียว — นี่เป็นหลักฐานที่แข็งที่สุดที่มีว่า in-dataset F1 ~0.9996 "
    "<b>ไม่ใช่</b> ข้อกล่าวอ้างเรื่องการใช้งานบน traffic ใหม่ ผลพังข้าม dataset "
    "เป็นสิ่งที่ต้องรายงานตรงๆ ไม่ใช่การทดลองที่ล้มเหลว",
], border=ACCENT)
note("ข้อจำกัด: วัดได้เฉพาะ Random Forest เพราะ behavioral features 12 ตัวต้องใช้ Source IP / Timestamp "
     "ซึ่ง CIC-UNSW-NB15 ไม่มี — ผลนี้ไม่บอกอะไรเกี่ยวกับ generalizability ของ cascade ที่ใช้จริง "
     "และต้องไม่รายงานเป็นผลของ ensemble")

# ══════════════════════════════════════════════════════════════════════
A(PageBreak())
h1("7. ส่วนที่วัดเพิ่มครบในรอบนี้")

h2("MTTD (M1b) — ตัวเลขหลักใช้ไม่ได้ และบอกไว้ตรงๆ")
p("ค่า MTTD ต่อ campaign ออกมา <b>0.0000 วินาที</b> ซึ่ง<b>ไม่ใช่ผลที่ดี แต่เป็นตัวเลขที่ไม่มีความหมาย</b> — "
  "flow แรกของทั้ง 228 campaign ถูกตรวจจับได้ทุกอัน เพราะ flow แรกมี behavioral vector เป็นศูนย์ทั้ง 12 คอลัมน์ "
  "(ถูกต้องทางกายภาพ: source IP ยังไม่มี activity ใน window) และกฎให้คะแนนเฉลี่ย 8.94 เทียบ threshold 0.03")
p("ตัวที่<b>อ้างได้</b> คือค่าที่ตัด flow แรกของแต่ละ campaign ออก:")
table(
    ["Attack class", "campaigns", "MTTD เฉลี่ย", "median", "max"],
    [
        ["port_scan", "4", "0.00 s", "0.00 s", "0.01 s"],
        ["credential_stuffing", "6", "0.14 s", "0.15 s", "0.31 s"],
        ["brute_force", "10", "0.98 s", "0.53 s", "2.63 s"],
        ["c2_beacon", "34", "193.27 s", "204.15 s", "482.94 s"],
        ["data_exfil", "174", "451.53 s", "417.41 s", "888.97 s"],
    ],
    [46 * mm, 24 * mm, 32 * mm, 30 * mm, 30 * mm],
)
note("รูปร่างสมเหตุสมผล: burst จับได้ทันที low-and-slow ใช้ 3–7 นาที · ไม่มี campaign ไหนหลุด · "
     "แต่ห้ามอ่านอัตราการตรวจจับบน overlay นี้เป็นหลักฐานความสามารถ เพราะ behavioral columns ของ overlay "
     "อ่อนกว่าข้อมูลจริงมาก (conn_count_1m สูงสุด 0.014 เทียบ 1.0 ในข้อมูลจริง)")

h2("SHAP — อธิบายได้ครั้งแรกเพราะการแก้ชั้นข้อมูล")
p("SHAP attribute <b>เฉพาะ branch ที่ตัดสินแถวนั้นจริง</b> โดยแถวที่ fallback ตัดสินจะ attribute "
  "ฟังก์ชันผสม <font face='Courier'>(if_norm + ae_norm)/2</font> ตรงๆ ซึ่ง<b>ทำได้เพราะทั้งสามโมเดล "
  "ใช้ feature space เดียวกันแล้ว</b> — ใต้ดีไซน์เดิมที่แยกสองชุด การเฉลี่ยคะแนนข้าม space "
  "ที่เข้ากันไม่ได้ไม่มีความหมายในการ attribute")
bullets([
    "Faithfulness (|base + Σshap − คะแนนจริง|): RF <b>1.96e-10</b> · fallback <b>4.66e-09</b>",
    "Latency: RF 0.258 s/alert · fallback 0.184 s — ใช้ได้ต่อ alert ไม่เหมาะกับการ score จำนวนมาก",
    "<b>ยืนยัน ablation โดยอิสระ:</b> feature อันดับต้นของ RF กลายเป็น behavioral columns "
    "(uniq_dport_count_5m 0.0517 · uniq_dport_count_3m 0.0417) โดย Destination Port ลงไปอันดับ 3",
])

h2("Branch mix — เหตุผลที่ cascade มีคุณค่าซึ่งตาราง LOAO มองไม่เห็น")
box([
    "ผลที่น่าเก็บไว้",
    "ใน dataset ปกติ RF ตัดสิน <b>99.785%</b> ของแถว fallback เพียง 0.215% — "
    "แต่บน synthetic overlay ซึ่งเป็น input นอก distribution "
    "<b>RF งดออกเสียงทุกแถว และ fallback ตัดสิน 100% ของ alert</b> "
    "นี่คือทางรองรับทำงานตามที่มันมีอยู่เพื่อสิ่งนี้ และเป็นข้อโต้แย้งให้ cascade เหนือ RF ตัวเดียว",
], border=GOOD, bg=colors.HexColor("#f0fdf4"))

# ══════════════════════════════════════════════════════════════════════
A(PageBreak())
h1("8. ข้อจำกัดที่ต้องเขียนในเล่ม")

p("รายการนี้บันทึกไว้ใน <font face='Leelawadee-Bold'>models/fusion_config.json</font> "
  "ไม่ใช่เขียนขึ้นใหม่ตอนทำรายงาน")

table(
    ["ข้อจำกัด", "ตัวเลข / สภาพจริง"],
    [
        ["hybrid แพ้ RF ตัวเดียวใน dataset",
         "F1 0.9966 เทียบ 0.9995 · FPR แย่กว่า 10 เท่า — เลือก hybrid เพราะชนะ zero-day"],
        ["Bot ตรวจไม่ได้เลย",
         "nested recall 0.000 · ทุกสมาชิกเกือบศูนย์ ไม่มีกฎรวมใดกู้คืนได้"],
        ["ตัด IF ออกจากการตัดสินมีต้นทุนจริง",
         "IF เป็นตัวที่ดีที่สุดบน DoS Slowhttptest (0.839 เทียบ AE 0.241, RF 0.055) แต่ FPR 0.09–0.15"],
        ["สองหมวดไม่มี redundancy",
         "FTP-Patator และ Web Attack - Brute Force ขึ้นกับ RF ตัวเดียว · PortScan ขึ้นกับ AE ตัวเดียว"],
        ["in-dataset metric อิ่มตัว",
         "RF val F1 0.9997 — แยกโมเดลไม่ออก ต้องใช้ zero-day LOAO เท่านั้น"],
        ["M1c วัดได้แค่ RF",
         "behavioral features ต้องใช้ IP/Timestamp ซึ่ง CIC-UNSW-NB15 ไม่มี"],
        ["MTTD ตัวเลขหลัก degenerate",
         "ต้องอ้างค่าที่ตัด flow แรกออก และบอกเหตุผลที่ตัด"],
        ["LLM layer ยังไม่ผ่านการตรวจคุณภาพ",
         "ยืนยันผลผลิตบน alert จริงไม่เกิน 2 เคส"],
        ["PA-5220 log จริงไม่มีและไม่ใช้",
         "อยู่นอกขอบเขตถาวร (ตัดสินใจ 25 ต.ค. 2569) — CICIDS-2017 เป็นชุดเทรนชุดเดียว"],
    ],
    [52 * mm, 110 * mm],
)

h1("9. สิ่งที่ยังไม่ได้ทำ")
bullets([
    "A/B analyst triage-time study (≥15 analysts × 20 alerts/group, blinded) — ยังไม่ได้เก็บข้อมูล",
    "LINE / Gmail ส่งจริง — ยังเป็น dry-run (sandbox ไม่มี outbound internet · Gmail OAuth2 เว้นว่างไว้)",
    "Kafka broker จริงและ OpenCTI จริงกับ enrich.py — ผ่าน 27 tests บน stub แต่ยังไม่เคยต่อของจริง",
    "ตรวจคุณภาพผลผลิต LLM บน alert จำนวนมากพอ",
])

gap(10)
note("ไฟล์อ้างอิงทั้งหมดอยู่ใน E:/thesis/reports/ — unified_val_metrics.csv · "
     "fusion_layer_loao_unified_or_comparison.csv · rf_behavioral_ablation_loao_report.txt · "
     "baseline_model_comparison.csv · siem_baseline_report.txt · cross_dataset_m1c_metrics.csv · "
     "mttd_metrics_excluding_first_flow.csv · shap_explainability_report.txt")

# ══════════════════════════════════════════════════════════════════════
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Leelawadee", 8.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(24 * mm, 12 * mm, "Hybrid IDS — การตอบคอมเมนต์อาจารย์ที่ปรึกษา · 1 พ.ย. 2569")
    canvas.drawRightString(186 * mm, 12 * mm, f"หน้า {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(24 * mm, 16 * mm, 186 * mm, 16 * mm)
    canvas.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4,
                      leftMargin=24 * mm, rightMargin=24 * mm,
                      topMargin=18 * mm, bottomMargin=22 * mm,
                      title="การตอบคอมเมนต์อาจารย์ที่ปรึกษา — Hybrid IDS",
                      author="Teerapong P.")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
doc.build(flow)
print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
