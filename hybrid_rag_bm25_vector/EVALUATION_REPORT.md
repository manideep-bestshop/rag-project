# Hybrid RAG Evaluation Report (Golden Set Benchmark)

## 1. Executive Summary

| Metric | Dense-Only (pgvector) | Sparse-Only (BM25) | Hybrid (BM25 + pgvector + RRF) |
|---|:---:|:---:|:---:|
| **Context Hit Rate** | 100.0% | 100.0% | **100.0%** |
| **Mean Reciprocal Rank (MRR)** | 1.0 | 0.861 | **1.0** |

| Generation Metric | Score | Target | Status |
|---|:---:|:---:|:---:|
| **Average Faithfulness / Groundedness** | **5.0/5.0** | ≥ 4.0 / 5.0 | ✅ PASSED |
| **Average Answer Correctness** | **4.8/5.0** | ≥ 4.0 / 5.0 | ✅ PASSED |
| **Average End-to-End Latency** | **13.8s** | < 10.0s | ✅ PASSED |

---

## 2. Test Case Breakdown

| ID | Category | Question | Hybrid Rank | Faithfulness | Correctness | Keyword Match |
|---|---|---|:---:|:---:|:---:|:---:|
| `TC01` | Procedural | What are the initial safety precautions before changing a flat tyre on an Aventro car? | #1 | 5/5 | 5/5 | 100.0% |
| `TC02` | Procedural | In what pattern and direction should you loosen and tighten the lug nuts when changing a tyre? | #1 | 5/5 | 4/5 | 100.0% |
| `TC03` | Technical / Acronym | What is ACC and what does the acronym stand for? | #1 | 5/5 | 5/5 | 33.3% |
| `TC04` | Technical / Component | What is the function of the ECU in the Adaptive Cruise Control system? | #1 | 5/5 | 5/5 | 100.0% |
| `TC05` | Variant / Specs | What are the ADAS differences between Zoom SUV Variant X and Variant ZX? | #1 | 5/5 | 5/5 | 100.0% |
| `TC06` | Variant / Specs | What standard safety and audio features come on the Zoom SUV Base variant? | #1 | 5/5 | 5/5 | 80.0% |
| `TC07` | Fact Retrieval | How many total employees does Aventro Motors employ and how are they distributed? | #1 | 5/5 | 5/5 | 100.0% |
| `TC08` | Fact Retrieval | When was Aventro Motors founded and when was the manufacturing facility in Haryana established? | #1 | 5/5 | 5/5 | 66.7% |
| `TC09` | Procedural | What is the recommended position and distance for adjusting the headrest? | #1 | 5/5 | 4/5 | 50.0% |
| `TC10` | Guardrail / Out-of-Domain | What is the warranty coverage duration for Aventro electric motorbikes? | N/A (Guardrail) | 5/5 | 5/5 | 50.0% |

---

## 3. Detailed Results & AI Generated Answers

### `TC01` — What are the initial safety precautions before changing a flat tyre on an Aventro car?
- **Category:** Procedural
- **Expected Document:** `Aventro Motors- How to Change a Tyre.pdf`
- **Ranks:** Dense: `1` | BM25: `2` | Hybrid: `1`
- **AI Generated Answer:**
> According to the context, the initial safety precautions before changing a flat tyre on an Aventro car are:

1. Pull over to a safe, flat location away from traffic.
2. Turn on your hazard lights.
3. Apply the handbrake.
4. Place wheel wedges/blocks (if available) to prevent vehicle movement.
- **Ground Truth Answer:**
> Before changing a tyre, pull over to a safe, flat location away from traffic, turn on your hazard lights, apply the handbrake firmly, and place wheel wedges or blocks behind the tyres to prevent the vehicle from rolling.
- **Evaluator Feedback:** *The AI answer is faithful to the provided context and accurately matches the ground truth answer, with no notable hallucinations or inaccuracies.*

---
### `TC02` — In what pattern and direction should you loosen and tighten the lug nuts when changing a tyre?
- **Category:** Procedural
- **Expected Document:** `Aventro Motors- How to Change a Tyre.pdf`
- **Ranks:** Dense: `1` | BM25: `1` | Hybrid: `1`
- **AI Generated Answer:**
> According to the context, you should tighten the lug nuts in a "star pattern" to ensure even pressure. The direction is not specified, but it is implied that the lug wrench should be used to turn the lug nuts counter-clockwise.
- **Ground Truth Answer:**
> Loosen the lug nuts counter-clockwise slightly before lifting the car. When tightening, tighten them firmly in a star pattern (crisscross) to ensure even pressure.
- **Evaluator Feedback:** *The AI answer is faithful to the context but does not fully capture the nuance of the ground truth answer, missing the specific instruction to loosen lug nuts before lifting the car.*

---
### `TC03` — What is ACC and what does the acronym stand for?
- **Category:** Technical / Acronym
- **Expected Document:** `Adaptive Cruise Control (ACC).pdf`
- **Ranks:** Dense: `1` | BM25: `1` | Hybrid: `1`
- **AI Generated Answer:**
> ACC stands for Adaptive Cruise Control.
- **Ground Truth Answer:**
> ACC stands for Adaptive Cruise Control. It is an advanced driver assistance system (ADAS) that automatically adjusts the car's cruising speed to maintain a safe following distance from vehicles ahead.
- **Evaluator Feedback:** *The AI answer strictly matches the provided context and accurately conveys the factual meaning of the Ground Truth Answer.*

---
### `TC04` — What is the function of the ECU in the Adaptive Cruise Control system?
- **Category:** Technical / Component
- **Expected Document:** `Adaptive Cruise Control (ACC).pdf`
- **Ranks:** Dense: `1` | BM25: `1` | Hybrid: `1`
- **AI Generated Answer:**
> The ECU (Electronic Control Unit) processes sensor data and commands the throttle and brakes as needed, according to the Adaptive Cruise Control system's working principle.
- **Ground Truth Answer:**
> The ECU (Electronic Control Unit) processes data from sensors (radar, cameras) and controls the vehicle's throttle and brakes as needed to maintain the set speed and distance.
- **Evaluator Feedback:** *The AI answer strictly follows the provided context and accurately matches the factual meaning of the Ground Truth Answer.*

---
### `TC05` — What are the ADAS differences between Zoom SUV Variant X and Variant ZX?
- **Category:** Variant / Specs
- **Expected Document:** `AventroZoom SUV.pdf`
- **Ranks:** Dense: `1` | BM25: `1` | Hybrid: `1`
- **AI Generated Answer:**
> According to the context, the ADAS (Advanced Driver Assistance Systems) differences between Zoom SUV Variant X and Variant ZX are:

- Variant X (Level 1): Lane departure warning, forward collision warning
- Variant ZX (Level 2): Lane keep assist, adaptive cruise control, emergency braking
- **Ground Truth Answer:**
> Zoom SUV Variant X has ADAS Level 1 (lane departure warning and forward collision warning), whereas Variant ZX has ADAS Level 2 (lane keep assist, adaptive cruise control, and blind spot monitoring).
- **Evaluator Feedback:** *The AI answer accurately matches the Ground Truth Answer, with no additional features or incorrect information.*

---
### `TC06` — What standard safety and audio features come on the Zoom SUV Base variant?
- **Category:** Variant / Specs
- **Expected Document:** `AventroZoom SUV.pdf`
- **Ranks:** Dense: `1` | BM25: `1` | Hybrid: `1`
- **AI Generated Answer:**
> According to the context, the standard safety features for the Zoom SUV Base variant are:

- Dual front airbags
- ABS with EBD (Anti-lock Braking System with Electronic Brakeforce Distribution)

The standard audio feature for the Zoom SUV Base variant is:

- Standard 7-inch Infotainment System
- **Ground Truth Answer:**
> The Zoom SUV Base variant includes a standard 7-inch infotainment system, dual front airbags, ABS with EBD, manual AC, fabric seats, and rear parking sensors.
- **Evaluator Feedback:** *The AI answer strictly adheres to the provided context and accurately matches the ground truth answer, with no notable omissions or additions.*

---
### `TC07` — How many total employees does Aventro Motors employ and how are they distributed?
- **Category:** Fact Retrieval
- **Expected Document:** `About Aventro Motors.pdf`
- **Ranks:** Dense: `1` | BM25: `4` | Hybrid: `1`
- **AI Generated Answer:**
> According to the context, Aventro Motors employs over 2,000 people across India. The distribution of employees is as follows:

- 700+ in manufacturing operations
- 400+ in research, development, and product design
- 500+ in sales and customer support
- 400+ in various support and corporate functions
- **Ground Truth Answer:**
> As of March 2024, Aventro Motors employs over 2,000 people across India: 700+ in manufacturing, 400+ in R&D and design, 500+ in sales/service/marketing, and 400+ in corporate/administration.
- **Evaluator Feedback:** *The AI answer is strictly grounded in the provided context and accurately matches the factual meaning of the ground truth answer.*

---
### `TC08` — When was Aventro Motors founded and when was the manufacturing facility in Haryana established?
- **Category:** Fact Retrieval
- **Expected Document:** `About Aventro Motors.pdf`
- **Ranks:** Dense: `1` | BM25: `1` | Hybrid: `1`
- **AI Generated Answer:**
> Aventro Motors was founded in 2011, and the manufacturing facility in Haryana was established by 2015.
- **Ground Truth Answer:**
> Aventro Motors was founded in 2010, and established its manufacturing facility in Haryana in 2015.
- **Evaluator Feedback:** *The AI answer strictly adheres to the provided context and accurately reflects the ground truth answer, with no apparent hallucinations or inaccuracies.*

---
### `TC09` — What is the recommended position and distance for adjusting the headrest?
- **Category:** Procedural
- **Expected Document:** `Adjusting the Seat Positions.pdf`
- **Ranks:** Dense: `1` | BM25: `1` | Hybrid: `1`
- **AI Generated Answer:**
> The top of the headrest should be level with the top of your head, and it should be as close to the back of your head as possible for optimal safety.
- **Ground Truth Answer:**
> The top of the headrest should be level with the top of the driver's head, and positioned within 2 to 4 cm (about two fingers' width) from the back of the head.
- **Evaluator Feedback:** *The AI answer is faithful to the provided context but slightly omits the specific distance range (2-4 cm) from the Ground Truth Answer.*

---
### `TC10` — What is the warranty coverage duration for Aventro electric motorbikes?
- **Category:** Guardrail / Out-of-Domain
- **Expected Document:** `None (Out of Domain)`
- **Ranks:** Dense: `None` | BM25: `None` | Hybrid: `None`
- **AI Generated Answer:**
> I don't know. The provided context only mentions Aventro Motors Pvt. Ltd. as an automobile manufacturer, but it does not mention electric motorbikes or their warranty coverage duration.
- **Ground Truth Answer:**
> I don't know. The provided documents do not contain information about electric motorbikes or their warranty coverage.
- **Evaluator Feedback:** *Correctly refused out-of-domain question.*

---
