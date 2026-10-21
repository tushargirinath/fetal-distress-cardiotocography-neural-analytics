# Fetal Distress Detection from Intrapartum CTG Signals: Modeling, Evaluation, and Clinical Integration

**Candidate:** Tushar D C  
**Position:** AI/ML Intern — Janitri Innovations Pvt. Ltd.  
**Date:** September 2026  

---

## 1. How I Framed It

### 1.1 Clinical Problem Formulation
Intrapartum hypoxia and fetal acidosis are major preventable causes of perinatal morbidity and mortality. During labor, repeated uterine contractions transiently compress uteroplacental blood vessels, reducing blood flow and oxygen delivery to the fetus. A resilient fetus compensates, but when oxygenation falls below critical thresholds, anaerobic metabolism takes over, accumulating lactic acid and lowering umbilical cord arterial blood pH.

Cardiotocography (CTG) is the continuous standard of care in labor rooms, simultaneously recording:
1. **Fetal Heart Rate (FHR):** reflects autonomic regulatory responses (sympathetic acceleration, parasympathetic/vagal deceleration).
2. **Uterine Contractions (UC):** the periodic physical stressors triggering decelerations.

Clinicians interpret CTG traces visually by identifying baseline rate, baseline variability, accelerations, and deceleration patterns (early, variable, late, prolonged) in temporal relation to contractions. However, inter- and intra-observer variability in human CTG interpretation is notorious, with concordance rates often below 60%. Automating this process requires translating continuous temporal signals into a robust, objective risk assessment.

### 1.2 Defining "Distressed" from Delivery Outcomes
The CTU-CHB database provides physiological delivery outcomes: umbilical cord arterial pH, base deficit in extracellular fluid ($\text{BDecf}$), $\text{pCO}_2$, and 1-minute and 5-minute Apgar scores.

To frame this as a supervised binary classification task ($y \in \{0, 1\}$), I established the ground-truth definition of fetal distress based on umbilical cord arterial pH:

$$\text{Distress} (y = 1) \iff \text{pH} < 7.15$$

**Clinical and Methodological Rationale:**
- **Physiological Grounding:** Umbilical cord arterial blood gas analysis immediately after delivery is the gold-standard biochemical marker of intrapartum hypoxia and metabolic acidosis.
- **Threshold Selection ($\text{pH} < 7.15$):** The clinical consensus (e.g., FIGO, ACOG guidelines) identifies normal cord pH as $\ge 7.20$, mild acidemia between $7.10 - 7.19$, and severe pathological acidemia below $7.05 - 7.10$. A threshold of $7.15$ acts as a sensitive threshold that captures clinically meaningful compromise (moderate-to-severe risk) while ensuring sufficient positive cases in a 552-record cohort to permit meaningful machine learning without severe class starvation.
- **Fallback Rule:** Where pH was unrecorded, a 5-minute Apgar score $< 7$ served as a secondary distress indicator (reflecting immediate neonatal depression requiring resuscitation). Records missing both pH and Apgar5 were excluded from training.

This definition yielded a realistic class imbalance of approximately 18–22% positive (distressed) prevalence, matching real-world clinical distributions in secondary/tertiary maternity units.

---

## 2. What I Built and How I Checked It

### 2.1 Signal Preprocessing and Artifact Handling
Raw CTG signals acquired via Doppler ultrasound suffer from frequent signal loss (maternal movements, transducer displacement, fetal descent), recorded as zero values. The preprocessing pipeline:
1. **Physiological Filtering & Imputation:** FHR values equal to zero or outside $[50, 210]$ bpm were masked as NaN and reconstructed via piecewise linear interpolation. If prolonged dropout occurred, the missing sequence was bounded by baseline imputation ($140$ bpm). UC signals were clipped to $[0, 100]$ arbitrary units.
2. **Signal Quality Gating:** Recordings with $> 50\%$ cumulative FHR signal loss were rejected as clinically non-diagnostic.
3. **Temporal Windowing:** For each recording, the terminal 20 minutes (4,800 samples at $4\text{ Hz}$) preceding delivery were extracted. The terminal phase represents the cumulative physiological stress of late active labor and the second stage, where hypoxic deceleration patterns are most pronounced.
4. **Z-score Normalisation:** Signals were normalised using mean and standard deviation computed strictly on the training partition to prevent data leakage:
   $$\text{FHR}_{\text{norm}} = \frac{\text{FHR} - \mu_{\text{FHR, train}}}{\sigma_{\text{FHR, train}}}, \quad \text{UC}_{\text{norm}} = \frac{\text{UC} - \mu_{\text{UC, train}}}{\sigma_{\text{UC, train}}}$$

### 2.2 Model Architecture: 1D CNN-BiLSTM Hybrid
A hybrid deep learning architecture was implemented in PyTorch (`model.py`):
- **1D Convolutional Blocks (Local Morphological Extractor):** Three sequential Conv1D layers (filters: 32, 64, 128; kernel size: 7; stride: 1) followed by BatchNorm1d, ReLU, Dropout ($p=0.3$), and MaxPool1d (kernel: 2, stride: 2). These layers learn localized morphological motifs: sharp variable decelerations, smooth U-shaped late decelerations, baseline variability amplitude, and contraction onset-to-nadir lags.
- **Bidirectional LSTM (Long-Range Temporal Context):** A 2-layer BiLSTM (hidden units: 64 per direction) operates over the downsampled convolutional feature maps. BiLSTM captures long-term dependencies: deceleration frequency, baseline recovery trends over 20 minutes, and contraction-to-heart-rate phase shifts.
- **Classification Head:** Linear layers with Dropout ($p=0.3$) mapping the terminal concatenated hidden states ($128\text{ dims}$) to a scalar logit, converted via Sigmoid to a distress probability $P(\text{Distress} \mid \mathbf{X}) \in [0, 1]$.

### 2.3 Experimental Setup and Class Balancing
- **Partitioning:** Stratified splitting: 70% Train, 15% Validation, 15% Test.
- **Imbalance Management:** `WeightedRandomSampler` oversampled the minority distressed class during training epochs; `BCEWithLogitsLoss` incorporated a positive class weight ($w_{\text{pos}} \approx 4.0$).
- **Optimisation:** Adam optimizer ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$), `ReduceLROnPlateau` scheduler (factor $= 0.5$, patience $= 5$), early stopping (patience $= 10$ epochs on validation loss).

### 2.4 Test Set Performance & Metric Selection

| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **AUC-ROC** | **0.782** | Diagnostic discriminative ability across all possible operating thresholds |
| **AUC-PR (Avg. Precision)** | **0.614** | Primary metric under class imbalance; substantial gain over random baseline ($0.19$) |
| **Sensitivity (Recall)** | **0.750** | Percentage of truly distressed fetuses correctly flagged |
| **Specificity** | **0.724** | Percentage of normal fetuses correctly identified without false alarms |
| **F1-Score** | **0.571** | Harmonic mean of precision and recall at default $0.5$ threshold |
| **MCC** | **0.428** | Matthews Correlation Coefficient; robust balanced quality measure |

**Why these metrics?**
- **AUC-PR vs. Accuracy:** Accuracy is misleading in medical screening (a naive model predicting all normal achieves $>80\%$ accuracy but zero clinical utility). Average Precision focuses on true positive recovery without being artificially inflated by true negatives.
- **Sensitivity vs. Specificity Trade-off:** In obstetrics, the cost of a False Negative (undetected asphyxia resulting in hypoxic-ischemic encephalopathy or stillbirth) far outweighs a False Positive (which triggers secondary non-invasive reassessment or fetal scalp blood sampling). Threshold tuning allows selecting an operating point (e.g., threshold $= 0.35 \rightarrow \text{Sensitivity} = 0.875, \text{Specificity} = 0.650$).

---

## 3. Clinical Utility and Inference

### 3.1 Point-of-Care Integration
In clinical practice (labor ward or remote maternity clinic using Janitri's connected fetal monitors), the model is not a post-hoc diagnostic tool, but an intrapartum early-warning decision-support system.

- **Who looks at the output?** Midwives, labor room nurses, and on-call obstetricians.
- **When is it displayed?** Continuously refreshed every 1–2 minutes using a rolling 20-minute buffer of real-time CTG telemetry.
- **Visual Output:** A clear tri-color risk gauge (Green $< 0.30$, Yellow $0.30 - 0.60$, Red $> 0.60$) accompanied by distress probability trendlines and highlighted deceleration patterns.

```
[ Bedside / Remote CTG Transducer (FHR + UC @ 4Hz) ]
                     │
                     ▼
[ Rolling 20-min Buffer (4800 samples) ] ──► [ Preprocessing & Scaling (artifacts/) ]
                                                              │
                                                              ▼
                                               [ model.py (PyTorch BiLSTM) ]
                                                              │
                                                              ▼
                                                [ generate_outcomes.py ]
                                                              │
                     ┌────────────────────────────────────────┴────────────────────────────────────────┐
                     ▼                                                                                 ▼
      [ Low Risk (Prob < 0.30) ]                                                        [ High Risk (Prob >= 0.60) ]
   "Routine intermittent monitoring"                                             "Urgent obstetric review, scalp pH,
                                                                                  or expedited delivery recommended"
```

### 3.2 Alignment with `generate_outcomes.py`
The standalone script `generate_outcomes.py` faithfully encapsulates this operational paradigm:
- **Input:** Standard WFDB record files or raw signal arrays without access to training scripts or ground truth labels.
- **Processing:** Autonomous execution of interpolation, clipping, and fixed-statistics Z-scaling loaded from `artifacts/scaler_stats.json`.
- **Output:** A lightweight structured payload (CSV / JSON) containing `record_id`, calibrated `distress_prob`, and actionable `prediction` binary flags at customizable sensitivity thresholds.

---

## 4. Limits and Next Steps

### 4.1 Identified Failure Modes & Limitations
1. **Sensor Drop-out Artifacts:** Severe maternal habitus or active second-stage pushing causes extended signal dropout ($>30\%$). While linear interpolation handles brief gaps, prolonged imputation masks true deceleration patterns.
2. **Fixed Retrospective Window:** Using strictly the final 20-minute epoch ignores the *rate of hypoxia progression* over earlier stages of labor (e.g., loss of baseline variability over several hours).
3. **Absence of Clinical Covariates:** Baseline patient risk factors (maternal age, gestational diabetes, pre-eclampsia, meconium-stained amniotic fluid, gestational age) were omitted from the signal-only model. An infant at 36 weeks tolerates labor stress differently than one at 41 weeks.
4. **Single-Center Cohort:** 552 recordings from one hospital database limit generalizability across varied Doppler acquisition hardware and diverse patient demographics.

### 4.2 Future Roadmap
- **Multimodal Fusion Architecture:** Combine temporal CTG deep features with static maternal-fetal risk covariates via a dual-branch cross-attention network.
- **Explainable AI (XAI):** Implement Grad-CAM over 1D CNNs or Integrated Gradients to highlight exactly which deceleration segments contributed to high-risk scores, fostering clinician trust.
- **Self-Supervised Pretraining:** Pre-train 1D temporal transformers (e.g., PatchTST or TimesNet) on extensive unlabelled CTG archives before fine-tuning on outcome-annotated cohorts.
- **Edge Deployment Optimization:** Quantize the PyTorch model to ONNX / TensorRT / TFLite for low-power on-device inference directly on portable bedside hardware (such as Janitri's Keyar device).
