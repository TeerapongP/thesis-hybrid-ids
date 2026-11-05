# บทที่ 2 — รายการอ้างอิงที่ตรวจสอบแล้ว
> สถานะ: กำลังค้น (อัปเดตระหว่างทาง)

## 1. การทบทวนวรรณกรรมเชิงสำรวจ: Machine Learning / Deep Learning สำหรับ Network Intrusion Detection

- [VERIFIED] [PEER-REVIEWED] Ferrag, M. A., Maglaras, L., Moschoyiannis, S., Janicke, H. (2020). "Deep learning for cyber security intrusion detection: Approaches, datasets, and comparative study". Journal of Information Security and Applications, 50. DOI: 10.1016/j.jisa.2019.102419 — citations: 1278
  - contribution: The standard comparative survey of DL-based IDS across public datasets; it is the baseline citation for situating this thesis's three-member design among existing approaches and for justifying dataset choice.
- [VERIFIED] [PEER-REVIEWED] Liu, H., Lang, B. (2019). "Machine Learning and Deep Learning Methods for Intrusion Detection Systems: A Survey". Applied Sciences, 9(20), 4396. DOI: 10.3390/app9204396 — citations: 1004
  - contribution: Provides the supervised/unsupervised/hybrid taxonomy this thesis's architecture is described in, and documents that hybrid combinations are an established rather than novel family.
- [VERIFIED] [PEER-REVIEWED] Aldweesh, A., Derhab, A., Emam, A. Z. (2020). "Deep learning approaches for anomaly-based intrusion detection systems: A survey, taxonomy, and open issues". Knowledge-Based Systems, 189, 105124. DOI: 10.1016/j.knosys.2019.105124 — citations: 586
  - contribution: Its "open issues" section names exactly the gaps this thesis attacks — unknown-attack detection, dataset obsolescence and lack of explainability.
- [VERIFIED] [PEER-REVIEWED] Moustafa, N., Hu, J., Slay, J. (2019). "A holistic review of Network Anomaly Detection Systems: A comprehensive survey". Journal of Network and Computer Applications, 128, 33-55. DOI: 10.1016/J.JNCA.2018.12.006 — citations: 305
  - contribution: Written by the UNSW-NB15 author; reviews anomaly-detection evaluation practice and supports the argument that single-dataset evaluation is a known methodological weakness.
- [VERIFIED] [PEER-REVIEWED] He, K., Kim, D. D., Asghar, M. R. (2023). "Adversarial Machine Learning for Network Intrusion Detection Systems: A Comprehensive Survey". IEEE Communications Surveys and Tutorials, 25(1). DOI: 10.1109/COMST.2022.3233793 — citations: 433
  - contribution: Top-tier venue survey documenting how brittle learned NIDS decision boundaries are, which supports this thesis's reluctance to read a saturated in-dataset F1 of 0.9995 as a deployment claim.
- [VERIFIED] [PEER-REVIEWED] Thakkar, A., Lohiya, R. (2020). "A Review on Machine Learning and Deep Learning Perspectives of IDS for IoT: Recent Updates, Security Issues, and Challenges". Archives of Computational Methods in Engineering, 28, 3211-3243. DOI: 10.1007/s11831-020-09496-0 — citations: 335
  - contribution: Reviews the dataset/class-imbalance and feature-selection pitfalls of IDS learning pipelines, which this thesis's preprocessing and capped-SMOTE decisions respond to.

## 2. ชุดข้อมูล CICIDS-2017: ตัวชุดข้อมูลเอง และงานวิจารณ์ข้อบกพร่อง

- [VERIFIED] [PEER-REVIEWED] Sharafaldin, I., Habibi Lashkari, A., Ghorbani, A. A. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization". ICISSP 2018. DOI: 10.5220/0006639801080116 — citations: 5542
  - contribution: The dataset paper itself — the mandatory primary citation for the only training dataset this thesis uses, and the source of the attack taxonomy the per-category LOAO evaluation is organised around.
- [VERIFIED] [PEER-REVIEWED] Engelen, G., Rimmer, V., Joosen, W. (2021). "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study". IEEE Security and Privacy Workshops (SPW) 2021. DOI: 10.1109/SPW53761.2021.00009 — citations: 247
  - contribution: The single most important critique for this thesis — it documents mislabelled flows, duplicated records and CICFlowMeter feature-extraction defects in CICIDS-2017, and directly justifies the deduplication, label normalisation and sentinel-value handling decisions in the preprocessing pipeline.
- [VERIFIED] [PEER-REVIEWED] Liu, L., Engelen, G., Lynar, T., Essam, D., Joosen, W. (2022). "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018". IEEE Conference on Communications and Network Security (CNS) 2022. DOI: 10.1109/CNS56114.2022.9947235 — citations: 157
  - contribution: Quantifies how widespread the labelling and generation errors are and shows that correcting them changes reported detection performance, which is why this thesis treats a near-perfect in-dataset F1 as an artefact to be distrusted rather than a result.
- [VERIFIED] [PEER-REVIEWED] Flood, R., Engelen, G., Aspinall, D., Desmet, L. (2024). "Bad Design Smells in Benchmark NIDS Datasets". IEEE European Symposium on Security and Privacy (EuroS&P) 2024. DOI: 10.1109/EuroSP60621.2024.00042 — citations: 47
  - contribution: Identifies systematic design flaws — including shortcut features that let models memorise dataset-specific cues — which is the published grounding for this thesis's concern about `Destination Port` ranking highly in SHAP and about dropping `Source IP` from model input as disguised leakage.

## 3. Hybrid / Ensemble IDS: การรวมตัวตรวจจับแบบ supervised และ unsupervised, กฎการหลอมรวม (fusion), cascade

- [VERIFIED] [PEER-REVIEWED] Mirsky, Y., Doitshman, T., Elovici, Y., Shabtai, A. (2018). "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection". Network and Distributed System Security Symposium (NDSS) 2018. DOI: 10.14722/NDSS.2018.23204 (arXiv:1802.09089) — citations: 1076
  - contribution: The canonical ensemble-of-unsupervised-detectors NIDS at a top-4 security venue; it is the strongest precedent for combining multiple reconstruction-based detectors and the natural comparison point for this thesis's unsupervised fallback branch.
- [VERIFIED] [PEER-REVIEWED] Tama, B. A., Comuzzi, M., Rhee, K.-H. (2019). "TSE-IDS: A Two-Stage Classifier Ensemble for Intelligent Anomaly-Based Intrusion Detection System". IEEE Access, 7. DOI: 10.1109/ACCESS.2019.2928048 — citations: 316
  - contribution: A published two-stage/cascaded ensemble IDS, establishing that this thesis's RF-primary-then-fallback cascade belongs to a recognised architectural family rather than being ad hoc.
- [VERIFIED] [PEER-REVIEWED] Thockchom, N., Singh, M. M., Nandi, U. (2023). "A novel ensemble learning-based model for network intrusion detection". Complex & Intelligent Systems, 9. DOI: 10.1007/s40747-023-01013-7 — citations: 122
  - contribution: A recent, well-cited ensemble NIDS whose combination rule and reported gains give a contemporary benchmark against which this thesis's modest 0.45pp zero-day margin over Random Forest alone can be honestly positioned.
- [VERIFIED] [PEER-REVIEWED] Megantara, A. A., Ahmad, T. (2021). "A hybrid machine learning method for increasing the performance of network intrusion detection systems". Journal of Big Data, 8. DOI: 10.1186/s40537-021-00531-w — citations: 86
  - contribution: Explicitly hybridises unsupervised and supervised stages for NIDS, supporting the design rationale that a supervised classifier and an anomaly detector cover complementary failure modes.
- [VERIFIED] [PEER-REVIEWED] Alotaibi, F., Maffeis, S. (2024). "Mateen: Adaptive Ensemble Learning for Network Anomaly Detection". International Symposium on Research in Attacks, Intrusions and Defenses (RAID) 2024. DOI: 10.1145/3678890.3678901 — citations: 13
  - contribution: A top-tier-venue recent work on how ensemble members should be selected and weighted over time, which is the closest published treatment of this thesis's finding that one member (the Autoencoder) can be removed at essentially no cost while another (Isolation Forest) carries the entire ensemble margin.

## 4. Isolation Forest (ต้นฉบับ) และการตรวจจับความผิดปกติด้วย Autoencoder บนทราฟฟิกเครือข่าย

- [VERIFIED] [PEER-REVIEWED] Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). "Isolation Forest". IEEE International Conference on Data Mining (ICDM) 2008. DOI: 10.1109/ICDM.2008.17 — citations: 7312
  - contribution: The original algorithm paper for one of this thesis's three members; required for citing the isolation-depth scoring mechanism and the `contamination`/threshold semantics that the contamination sweep tunes.
- [VERIFIED] [PEER-REVIEWED] Liu, F. T., Ting, K. M., Zhou, Z.-H. (2012). "Isolation-Based Anomaly Detection". ACM Transactions on Knowledge Discovery from Data (TKDD), 6(1). DOI: 10.1145/2133360.2133363 — citations: 2093
  - contribution: The extended journal treatment, which is the better citation for why Isolation Forest's score is a ranking rather than a calibrated probability — exactly the property that forces this thesis to min-max normalise it before averaging it with the Autoencoder's error.
- [VERIFIED] [PEER-REVIEWED] Pang, G., Shen, C., Cao, L., Hengel, A. van den (2021). "Deep Learning for Anomaly Detection: A Review". ACM Computing Surveys, 54(2). DOI: 10.1145/3439950 (arXiv:2007.02500) — citations: 1566
  - contribution: The standard review of deep anomaly detection, which frames reconstruction-error methods (this thesis's Autoencoder) as one family among several and documents their known weakness on low-volume anomalies — matching the near-zero recall observed here on Bot and the Patator classes.
- [VERIFIED] [PEER-REVIEWED] Zavrak, S., İskefiyeli, M. (2020). "Anomaly-Based Intrusion Detection From Network Flow Features Using Variational Autoencoder". IEEE Access, 8. DOI: 10.1109/ACCESS.2020.3001350 — citations: 256
  - contribution: A directly comparable autoencoder-on-flow-features NIDS evaluated on CICIDS-2017, giving a published precedent for training the reconstruction model on benign traffic only and thresholding on a percentile of benign reconstruction error.
- [VERIFIED] [PEER-REVIEWED] Mirsky, Y., Doitshman, T., Elovici, Y., Shabtai, A. (2018). "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection". NDSS 2018. DOI: 10.14722/NDSS.2018.23204 — citations: 1076
  - contribution: (also cited under topic 3) The strongest security-venue precedent for autoencoder reconstruction error as an online NIDS signal, and for treating that error as an anomaly score rather than a classifier output.

## 10. ภาระงานนักวิเคราะห์ SOC: alert fatigue, ภาระ false positive, การศึกษาเวลา triage เชิงประจักษ์

- [UNVERIFIED-API / publisher-confirmed] [PEER-REVIEWED] Alahmadi, B. A., Axon, L., Martinovic, I. (2022). "99% False Positives: A Qualitative Study of SOC Analysts' Perspectives on Security Alarms". 31st USENIX Security Symposium, pp. 2783-2800. https://www.usenix.org/conference/usenixsecurity22/presentation/alahmadi
  - contribution: The definitive qualitative evidence that false-positive burden, not detection rate, is the dominant practical constraint in a real SOC — the core motivation for this thesis's emphasis on FPR and for adding an explanation layer at all. (Title, all three authors, venue and page range were read directly from the USENIX publisher page this session; the Semantic Scholar record could not be retrieved before the API rate limit, so the tag is conservative.)
- [UNVERIFIED-API / publisher-confirmed] [PEER-REVIEWED] Yang, L., Chen, Z., Wang, C., Zhang, Z., Booma, S., Cao, P., Adam, C., Withers, A., Kalbarczyk, Z., Iyer, R. K., Wang, G. (2024). "True Attacks, Attack Attempts, or Benign Triggers? An Empirical Measurement of Network Alerts in a Security Operations Center". 33rd USENIX Security Symposium, pp. 1525-1542. https://www.usenix.org/conference/usenixsecurity24/presentation/yang-limin
  - contribution: A four-year, 115-million-alert measurement study in a real university-affiliated SOC (NCSA) — the closest published analogue to this thesis's campus-SOC setting, and the strongest empirical grounding for why an extremely low FPR matters more than a marginal recall gain at campus alert volumes. (Verified the same way as the entry above.)

## 11. โมเดลภาษาขนาดใหญ่ภาษาไทย (ตระกูล Typhoon จาก SCB 10X) และการประเมิน LLM ที่ไม่ใช่ภาษาอังกฤษ

- [UNVERIFIED-API / arXiv-confirmed] [PREPRINT] Pipatanakul, K., Manakul, P., Nitarach, N., Sirichotedumrong, W., Nonesung, S., Jaknamon, T., Pengpun, P., Taveekitworachai, P., Na-Thalang, A., Sripaisarnmongkol, S., Jirayoot, K., Tharnpipitchai, K. (2024). "Typhoon 2: A Family of Open Text and Multimodal Thai Large Language Models". arXiv:2412.13702. DOI: 10.48550/arXiv.2412.13702 — https://arxiv.org/abs/2412.13702
  - contribution: The technical report for the exact model family this thesis deploys (Typhoon2 8B Instruct, Llama-3.1-based), and the only citable source for its training recipe, Thai benchmark scores and extended context length — i.e. the justification for choosing it over a general multilingual model for Thai analyst-facing output.
- [UNVERIFIED-API / arXiv-confirmed] [PREPRINT] Pipatanakul, K., Jirabovonvisut, P., Manakul, P., Sripaisarnmongkol, S., Patomwong, R., Chokchainant, P., Tharnpipitchai, K. (2023). "Typhoon: Thai Large Language Models". arXiv:2312.13951. DOI: 10.48550/arXiv.2312.13951 — https://arxiv.org/abs/2312.13951
  - contribution: The first-generation Typhoon report, which introduces the ThaiExam evaluation set and documents Thai tokenisation efficiency — the background needed to argue that Thai-language output quality must be evaluated with Thai-specific instruments rather than assumed from English benchmarks.

Both entries were confirmed by fetching the arXiv abstract pages directly this session (title, full author list, submission date and DOI all read from the publisher page). The Semantic Scholar identifier lookups for arXiv IDs were returning rate-limit errors during this session, so these two are tagged conservatively rather than as [VERIFIED]; the DOIs above are canonical arXiv DOIs and resolve.

## 9. Explainable AI สำหรับ IDS โดยเฉพาะการใช้ SHAP กับการตรวจจับการบุกรุก

- [VERIFIED] [PEER-REVIEWED] Lundberg, S. M., Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions". Advances in Neural Information Processing Systems (NeurIPS) 30. arXiv:1705.07874 — citations: 45153
  - contribution: The SHAP method paper itself; it supplies the additivity property (`base_value + sum(shap) = model output`) that this thesis exploits as a faithfulness check per fusion branch, and the `TreeExplainer`/`KernelExplainer` distinction behind attributing the RF branch exactly and the unsupervised fallback by sampling.
- [VERIFIED] [PEER-REVIEWED] Wang, M., Zheng, K., Yang, Y., Wang, X. (2020). "An Explainable Machine Learning Framework for Intrusion Detection Systems". IEEE Access, 8, 73127-73141. DOI: 10.1109/ACCESS.2020.2988359 — citations: 312
  - contribution: The most-cited application of SHAP to IDS and the standard reference for the local-plus-global explanation split that this thesis follows (per-alert attributions for the analyst, mean |SHAP| rankings for the global feature analysis).
- [VERIFIED] [PEER-REVIEWED] Zhang, Z., Al Hamadi, H., Damiani, E., Yeun, C. Y., Taher, F. (2022). "Explainable Artificial Intelligence Applications in Cyber Security: State-of-the-Art in Research". IEEE Access, 10. DOI: 10.1109/ACCESS.2022.3204051 (arXiv:2208.14937) — citations: 378
  - contribution: Surveys XAI in security broadly and documents that explanation latency and analyst-facing usefulness, not just attribution correctness, are the open problems — the framing for this thesis's measured per-alert SHAP cost (0.18-0.26 s) and its claim that the method is per-alert usable but impractical for bulk scoring.

