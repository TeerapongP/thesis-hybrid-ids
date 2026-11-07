# การทดสอบระบบแบบครบวง (Full-Loop Test)

> ขั้นตอนนี้ทดสอบ **สายที่วัดผลจริง: เมทริกซ์คุณลักษณะ → LINE** ไม่ใช่ตั้งแต่ ingestion
> ขา ingestion (`pa5220.raw`) ถูกตัดออกเมื่อ 2569-11-07 เพราะ log PA-5220 อยู่นอกขอบเขตถาวร
> ตามข้อเสนอข้อ 3.2 และเลือกไม่จำลอง feed ขึ้นมาแทน ดูเหตุผลใน CLAUDE.md
>
> ทุกคำสั่งในไฟล์นี้เคยรันจริงแล้วอย่างน้อยหนึ่งครั้ง ไม่ใช่คำสั่งที่เขียนจากการคาดเดา


> **ทางอัตโนมัติ** `src/e2e_replay.py` รันสายเดียวกันนี้ให้ต่อกันตั้งแต่ replay flow event เข้า Kafka
> ผ่าน `enrich.py` ที่ไม่ถูกแก้ ไปจนถึงให้คะแนน อธิบาย และ dispatch โดยไม่ต้องไล่ทำขั้น 6–7 ด้วยมือ
>
> ```bash
> uv run python -m src.e2e_replay --check
> uv run python -m src.services.enrich            # เปิดอีกหน้าต่าง runner ไม่เปิด/ปิดให้
> uv run python -m src.e2e_replay --label PortScan --dry-run
> ```
>
> ใช้ `--from <stage>` (`export` `produce` `consume` `score` `notify`) เพื่อลองสเตจท้าย ๆ ซ้ำโดยไม่ต้อง
> enrich ใหม่ **คู่มือด้านล่างยังคงไว้สำหรับตอนที่สเตจใดล้ม** เพราะบอกว่าต้องดูตรงไหน
>
> **สิ่งที่ runner พิสูจน์ได้คือ "end to end จาก flow event ที่ replay"** ไม่ใช่จาก ingestion หรือจาก
> traffic ดิบ เพราะ event แบกคุณลักษณะ 94 ตัวไปในตัว และใน repo นี้ไม่มีขั้นสกัดคุณลักษณะจาก log
> ดูรายละเอียดและข้อบกพร่องที่พบ (เช่น Logstash ปฏิเสธ event ที่มี `features` ด้วย HTTP 400)
> ใน CLAUDE.md หัวข้อ "Replay runner"
>
> ข้อสังเกตที่ยังไม่ยืนยัน: เงื่อนไขข้อ 5 ด้านล่างบอกว่ากิ่ง fallback ให้ MEDIUM เสมอ แต่ dry-run เมื่อ
> 2569-11-07 ได้ HIGH สำหรับแถว 215249 และ 24258 (ทั้งคู่ `if_ae_fallback`) และ MEDIUM สำหรับ 103508
> ยังไม่ได้ dispatch จริง จึงไม่รู้ว่า LINE เด้งหรือไม่ ให้ตรวจก่อนเชื่อข้อ 5

---

## 0. เงื่อนไขก่อนเริ่ม — หกข้อ ที่เคยพลาดมาแล้วทุกข้อ

| # | เงื่อนไข | ถ้าไม่ทำจะเกิดอะไร |
|---|---|---|
| 1 | Docker stack ทั้งสามขึ้น | ทุกอย่างหลังขั้น 3 ล้มหมด |
| 2 | **vhost `soc` ถูกสร้าง** | `dispatch_alert` ล้มด้วย `(530) NOT_ALLOWED` ขณะที่ InfluxDB/ES ผ่าน — อ่านเหมือนบั๊กในโค้ด |
| 3 | Ollama รันพร้อมโมเดล Typhoon2 | ชั้น LLM ล้ม ต้องใช้ `--no-llm` แทน |
| 4 | **ช่วงเวลา Grafana = ปี 2017** | แผงว่างเปล่า ทั้งที่ข้อมูลเข้าแล้ว |
| 5 | เลือกแถวที่ **RF ตัดสิน** ถ้าอยากให้ LINE เด้ง | กิ่ง fallback ให้ MEDIUM เสมอ → ไม่ส่ง LINE |
| 6 | อินเทอร์เน็ตขาออก | LINE ส่งไม่ได้ (แต่ `--dry-run` ไม่ต้องใช้) |

---

## 1. ทดสอบหน่วยก่อน — ถูกกว่าการไล่หาปัญหาตอนรันจริง

ชุดทดสอบ**แยกสองล่าม** รันด้วยล่ามเดียวจะเห็น failure ปลอม

```bash
python -m unittest discover -s tests -p "test_[!l]*.py"
```

```bash
uv run python -m unittest discover -s tests -p "test_llm*.py"
```

**ต้องได้ `OK` ทั้งสองชุด** ถ้าชุดแรกแดงด้วย `ModuleNotFoundError: pandas` แปลว่ารันผิดล่าม

---

## 2. ยกโครงสร้างพื้นฐาน

```bash
docker network create soc_shared
```

```bash
docker compose -f network/docker-compose.network.yml up -d
```

```bash
docker compose --env-file opencti/.env -f opencti/docker-compose.yml up -d
```

```bash
docker compose --env-file observability/.env -f observability/docker-compose.yml up -d
```

```bash
docker compose --env-file notification/.env -f grafana/docker-compose.yml up -d
```

รอจน healthcheck ผ่าน (~1–2 นาที) แล้วตรวจพอร์ต

```bash
python -c "import socket;[print(f'{n:9s}:{p:<6}', 'OPEN' if __import__('socket').socket().connect_ex(('127.0.0.1',p))==0 else 'closed') for n,p in (('kafka',9092),('opencti',8082),('influx',8086),('elastic',9200),('grafana',3000))]"
```

---

## 3. สร้าง vhost `soc` — ไม่มีอะไรในโปรเจคสร้างให้

image `rabbitmq` ไม่มี setting แบบ declarative สำหรับ vhost เลย Compose ทำแทนไม่ได้
**ต้องทำทุกครั้งที่สร้าง volume ใหม่**

```bash
docker exec opencti-opencti-rabbitmq-1 rabbitmqctl add_vhost soc
```

```bash
docker exec opencti-opencti-rabbitmq-1 rabbitmqctl set_permissions -p soc opencti ".*" ".*" ".*"
```

ตรวจว่าขึ้นจริง

```bash
docker exec opencti-opencti-rabbitmq-1 rabbitmqctl list_vhosts
```

---

## 4. เปิด Ollama

```bash
ollama serve
```

เช็กว่าโมเดลอยู่ (เปิดอีกหน้าต่าง)

```bash
curl -s http://127.0.0.1:11434/api/tags
```

**ต้องเห็น `scb10x/llama3.1-typhoon2-8b-instruct`** ถ้าไม่มีให้ `ollama pull` ก่อน

---

## 5. ตรวจช่องทาง LINE โดยยังไม่ส่ง

```bash
uv run python -m src.notifications.line_preflight
```

คาดหวัง — **ยังไม่ส่งข้อความใด ๆ ในขั้นนี้**

```
ok    token      configured (172 chars; value not printed)
ok    channel    "SOC Alert Lab" @360hvcni chatMode=bot
ok    quota      type=limited value=300
ok    mode       broadcast -> https://api.line.me/v2/bot/message/broadcast
PASS  channel is usable; send_line_alert will broadcast to all friends.
```

ถ้าได้ `FAIL token /v2/bot/info -> HTTP 401` แปลว่า token ถูกเพิกถอน ต้องออกใหม่ใน console

---

## 6. รันครบเจ็ดชั้นแบบไม่ส่งอะไร

`--dry-run` เป็นค่าเริ่มต้น จึงปลอดภัยเสมอ และ**ไม่ต้องใช้อินเทอร์เน็ต**

```bash
uv run python -m src.pipeline --label "DoS Hulk" --shap
```

ต้องเห็นครบทุกชั้นในผลลัพธ์

| ชั้น | สิ่งที่ต้องเห็น |
|---|---|
| RF / IF / AE | คะแนนทั้งสามตัว |
| FUSION | `decided_by=random_forest` หรือ `if_ae_fallback` |
| SHAP | `branch=` ตรงกับ `decided_by` และคุณลักษณะ 5 อันดับ |
| RAG | 3 chunk พร้อมคะแนน |
| LLM | คำอธิบายภาษาไทย + มาตรการ |
| DISPATCH | `DRY RUN -- nothing sent` |

**ทดสอบกิ่ง fallback ด้วย** เพราะ RF ตัดสิน 99.785% ของแถว ถ้าสุ่มจะไม่เจอกิ่งนี้เลย

```bash
uv run python -m src.pipeline --row 215249 --row 103508 --row 24258 --shap
```

สามแถวนี้ `decided_by=if_ae_fallback` และเป็น Heartbleed / Bot / DDoS

> **เลขแถวพวกนี้ผูกกับ split ปัจจุบัน** มาจากการสแกนจริงเมื่อ 2569-11-06 ถ้ามีการรัน
> `build_unified_population.py` ใหม่หรือเปลี่ยน seed เลขจะไม่ตรงอีก วิธีหาใหม่คือ
> ให้คะแนน test split ด้วย RF แล้วเลือกแถวที่ `|P(attack) − 0.5| × 2 < 0.95`
> ซึ่งเป็นเงื่อนไขเดียวกับที่ `fusion_layer.CONFIDENCE_THRESHOLD` ใช้

---

## 7. ส่งจริงหนึ่งครั้ง

> **ขั้นนี้ broadcast เข้า LINE ของทุกคนที่ add บอท และกินโควตาตามจำนวนผู้รับ**
> เลือกแถวที่ RF ตัดสิน ไม่งั้นได้ MEDIUM แล้ว LINE ไม่เด้ง

```bash
uv run python -m src.pipeline --label "DoS Hulk" --dispatch
```

ต้องได้

```
DISPATCH  alert_id=...  priority=HIGH  rabbitmq=ok  influxdb=ok  elasticsearch=ok  notification=sent
```

แล้วเช็กมือถือว่าข้อความเข้าจริง

---

## 8. ยืนยันว่าแดชบอร์ดเห็นข้อมูลที่เพิ่งเขียน

```bash
uv run python tests/verify_dashboard_queries.py
```

ต้องได้ `RESULT: PASS`

**ถ้าได้ FAIL ให้ดูสาเหตุก่อนสรุปว่าพัง** — สคริปต์นี้ query InfluxDB ช่วง `-30d` ถ้ายังไม่ได้ dispatch
อะไรเลย มันจะ FAIL เพราะหน้าต่างว่าง ไม่ใช่เพราะเส้นทางเสีย

---

## 9. ดูด้วยตาใน Grafana กับ Kibana

| เครื่องมือ | URL | สิ่งที่ต้องตั้งก่อน |
|---|---|---|
| Grafana | http://localhost:3000 | **ตั้งช่วงเวลาเป็นช่วงปี 2017** |
| Kibana | http://localhost:5601 | index pattern `soc-alerts-*` |

**กับดักที่ต้องรู้ และมันกระทบทั้งสองเครื่องมือ ไม่ใช่แค่ Grafana** เวลาของเหตุการณ์คือเวลา
ของ flow ซึ่ง replay จาก CICIDS-2017 คือ **2017-07-05** ไม่ใช่เวลาปัจจุบัน

| ที่ไหน | อาการ |
|---|---|
| Grafana | เปิดด้วยช่วง "Last 6 hours" แล้วว่างเปล่า ทั้งที่ `influxdb=ok` |
| Kibana | เอกสารไปอยู่ใน index **`soc-alerts-2017.07.05`** ไม่ใช่ index ของวันนี้ — `soc-alerts-<วันนี้>` ไม่มีอยู่เลย |

ยืนยันแล้ว 2569-11-07: alert ที่เพิ่ง dispatch อยู่ใน `soc-alerts-2017.07.05` จริง และไม่โผล่ใน
มุมมองที่กรองด้วยเวลาล่าสุดเลย — **`elasticsearch=ok` พูดความจริง**

**การค้นหา alert_id ใน Kibana มีกับดักของตัวเอง** ฟิลด์ `alert_id` ถูก map เป็น `text`
(มี subfield `.keyword`) ดังนั้น

```
term  alert_id         -> 0 hits     (UUID ถูกตัดเป็น token ด้วยขีด)
term  alert_id.keyword -> 1 hit      ใช้ตัวนี้สำหรับ filter แบบตรงตัว
match alert_id         -> 1 hit      ใช้ตัวนี้ในช่องค้นหาปกติ
```

ผมเองเพิ่งพลาดข้อนี้ระหว่างทดสอบ — ใช้ `term` แล้วได้ 0 hit และเกือบสรุปว่าการเขียนล้มเหลว

**ข้อจำกัดอีกข้อ** instance นี้คืนผลไม่สอดคล้องกันเมื่อ query ช่วงกว้าง (วัดแล้ว 2569-11-06
หาสาเหตุไม่เจอ) **ใช้ช่วงแคบเท่านั้น** อย่าอ่านช่วงหลายเดือนว่าครบ

---

## 10. องค์ประกอบที่อยู่นอกวง — ทดสอบแยก

`enrich.py` ไม่มีข้อมูลนำเข้าสดในงานนี้ จึงไม่อยู่ในวงข้างบน แต่ทดสอบตัวเองได้

```bash
uv run python -m src.services.enrich --dry-run
```

ต้องได้ `dry-run done: {'published': 3, 'skipped_malformed': 0}`

---

## 11. ปิดงาน

ปิดทีละ stack ด้วย env file ของตัวเอง — การรวมสาม compose ในคำสั่งเดียวโดยไม่ส่ง
`--env-file` ทำให้ตัวแปรของ OpenCTI ว่างทั้งหมดและขึ้น warning ยาวเหยียด

```bash
docker compose --env-file notification/.env -f grafana/docker-compose.yml down
```

```bash
docker compose --env-file observability/.env -f observability/docker-compose.yml down
```

```bash
docker compose --env-file opencti/.env -f opencti/docker-compose.yml down
```

> ใช้ `down` เฉย ๆ **ห้ามใส่ `-v`** — `-v` ลบ volume ทิ้ง ซึ่งหมายถึง vhost `soc`,
> ข้อมูล OpenCTI ที่ ingest มาแล้ว และ dashboard ของ Grafana หายทั้งหมด

---

## สรุปสิ่งที่วงนี้พิสูจน์ และไม่พิสูจน์

**พิสูจน์ได้** — แบบจำลองสามตัว กฎรวมผล SHAP RAG แบบจำลองภาษาไทย และชั้นแจ้งเตือน
ทำงานต่อกันได้จริงกับบริการจริงทุกตัว ไม่มีส่วนใดถูกแทนด้วยตัวทดสอบเทียม

**พิสูจน์ไม่ได้** — ความสามารถในการรับ log สดจากอุปกรณ์จริง (ขา ingestion อยู่นอกขอบเขต),
ความแม่นยำของการตรวจจับ (วัดแยกด้วย M1a/LOAO/M1c ไม่ใช่ด้วยการรันสาธิต), และ Gmail
(ยังไม่ตั้งค่า OAuth2)
