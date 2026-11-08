"""
Golden Set Generator for RAG Evaluation.
Extracts and curates a verified Ground Truth Benchmark (Golden Set) from the Aventro corpus.

Each test case contains:
  - id: Test case ID
  - category: Procedural, Technical/Acronym, Fact Retrieval, Variant/Spec, Guardrail
  - query: Realistic user question
  - source_file: Expected document
  - ground_truth_context: Key excerpt from the document
  - ground_truth_answer: Human/expert reference answer
  - expected_keywords: Key terms that must appear for correctness
"""

import json
import csv
import os
from pathlib import Path

GOLDEN_SET = [
    {
        "id": "TC01",
        "category": "Procedural",
        "source_file": " Aventro Motors- How to Change a Tyre.pdf",
        "query": "What are the initial safety precautions before changing a flat tyre on an Aventro car?",
        "ground_truth_context": (
            "1. Park Safely: Pull over to a safe, flat location away from traffic. "
            "Turn on your hazard lights. Apply the handbrake firmly. "
            "Place wheel wedges or blocks (if available) behind the tyres to prevent rolling."
        ),
        "ground_truth_answer": (
            "Before changing a tyre, pull over to a safe, flat location away from traffic, "
            "turn on your hazard lights, apply the handbrake firmly, and place wheel wedges or blocks "
            "behind the tyres to prevent the vehicle from rolling."
        ),
        "expected_keywords": ["hazard lights", "handbrake", "flat location", "wheel wedges"]
    },
    {
        "id": "TC02",
        "category": "Procedural",
        "source_file": " Aventro Motors- How to Change a Tyre.pdf",
        "query": "In what pattern and direction should you loosen and tighten the lug nuts when changing a tyre?",
        "ground_truth_context": (
            "Loosen the Lug Nuts: Use the lug wrench to turn the lug nuts counter-clockwise. "
            "Loosen them slightly – do not remove fully at this stage... "
            "Tighten the Lug Nuts: Use the lug wrench to tighten the lug nuts in a star pattern to ensure even pressure."
        ),
        "ground_truth_answer": (
            "Loosen the lug nuts counter-clockwise slightly before lifting the car. "
            "When tightening, tighten them firmly in a star pattern (crisscross) to ensure even pressure."
        ),
        "expected_keywords": ["counter-clockwise", "star pattern"]
    },
    {
        "id": "TC03",
        "category": "Technical / Acronym",
        "source_file": "Adaptive Cruise Control (ACC).pdf",
        "query": "What is ACC and what does the acronym stand for?",
        "ground_truth_context": (
            "Adaptive Cruise Control (ACC): What is it & How Does it Work? "
            "Overview: Adaptive Cruise Control (ACC) is an advanced driver assistance system (ADAS) "
            "designed to automatically adjust a vehicle's cruising speed to maintain a safe following distance."
        ),
        "ground_truth_answer": (
            "ACC stands for Adaptive Cruise Control. It is an advanced driver assistance system (ADAS) "
            "that automatically adjusts the car's cruising speed to maintain a safe following distance from vehicles ahead."
        ),
        "expected_keywords": ["Adaptive Cruise Control", "following distance", "speed"]
    },
    {
        "id": "TC04",
        "category": "Technical / Component",
        "source_file": "Adaptive Cruise Control (ACC).pdf",
        "query": "What is the function of the ECU in the Adaptive Cruise Control system?",
        "ground_truth_context": (
            "• ECU (Electronic Control Unit): Processes sensor data and commands the throttle and brakes as needed."
        ),
        "ground_truth_answer": (
            "The ECU (Electronic Control Unit) processes data from sensors (radar, cameras) and controls the vehicle's throttle and brakes as needed to maintain the set speed and distance."
        ),
        "expected_keywords": ["Electronic Control Unit", "throttle", "brakes", "sensor data"]
    },
    {
        "id": "TC05",
        "category": "Variant / Specs",
        "source_file": " AventroZoom SUV.pdf",
        "query": "What are the ADAS differences between Zoom SUV Variant X and Variant ZX?",
        "ground_truth_context": (
            "Key Features & Technology • ADAS (Advanced Driver Assistance Systems): "
            "– Level 1 (X): Lane departure warning, forward collision warning "
            "– Level 2 (ZX): Lane keep assist, adaptive cruise control, blind spot monitoring"
        ),
        "ground_truth_answer": (
            "Zoom SUV Variant X has ADAS Level 1 (lane departure warning and forward collision warning), "
            "whereas Variant ZX has ADAS Level 2 (lane keep assist, adaptive cruise control, and blind spot monitoring)."
        ),
        "expected_keywords": ["Level 1", "Level 2", "lane departure", "lane keep assist", "adaptive cruise control"]
    },
    {
        "id": "TC06",
        "category": "Variant / Specs",
        "source_file": " AventroZoom SUV.pdf",
        "query": "What standard safety and audio features come on the Zoom SUV Base variant?",
        "ground_truth_context": (
            "Zoom SUV Base: • Standard 7-inch Infotainment System • Dual front airbags, ABS with EBD • Manual AC • Fabric seats • Rear parking sensors"
        ),
        "ground_truth_answer": (
            "The Zoom SUV Base variant includes a standard 7-inch infotainment system, dual front airbags, ABS with EBD, manual AC, fabric seats, and rear parking sensors."
        ),
        "expected_keywords": ["7-inch", "airbags", "ABS", "EBD", "parking sensors"]
    },
    {
        "id": "TC07",
        "category": "Fact Retrieval",
        "source_file": "About Aventro Motors.pdf",
        "query": "How many total employees does Aventro Motors employ and how are they distributed?",
        "ground_truth_context": (
            "Employees: As of March 2024, Aventro Motors employs over 2,000 people across India. "
            "– 700+ in manufacturing operations. – 400+ in research, development, and product design. "
            "– 500+ in sales, service, and marketing. – 400+ in corporate, finance, and administrative functions."
        ),
        "ground_truth_answer": (
            "As of March 2024, Aventro Motors employs over 2,000 people across India: 700+ in manufacturing, "
            "400+ in R&D and design, 500+ in sales/service/marketing, and 400+ in corporate/administration."
        ),
        "expected_keywords": ["2,000", "700", "400", "500", "manufacturing"]
    },
    {
        "id": "TC08",
        "category": "Fact Retrieval",
        "source_file": "About Aventro Motors.pdf",
        "query": "When was Aventro Motors founded and when was the manufacturing facility in Haryana established?",
        "ground_truth_context": (
            "Aventro Motors was founded in 2010... By 2015, Aventro established its state-of-the-art manufacturing facility in Haryana."
        ),
        "ground_truth_answer": (
            "Aventro Motors was founded in 2010, and established its manufacturing facility in Haryana in 2015."
        ),
        "expected_keywords": ["2010", "2015", "Haryana"]
    },
    {
        "id": "TC09",
        "category": "Procedural",
        "source_file": "Adjusting the Seat Positions.pdf",
        "query": "What is the recommended position and distance for adjusting the headrest?",
        "ground_truth_context": (
            "The top of the headrest should be level with the top of your head. "
            "The headrest should be positioned close to the back of your head, ideally within 2 to 4 cm (about two fingers' width)."
        ),
        "ground_truth_answer": (
            "The top of the headrest should be level with the top of the driver's head, "
            "and positioned within 2 to 4 cm (about two fingers' width) from the back of the head."
        ),
        "expected_keywords": ["level with the top", "2 to 4 cm"]
    },
    {
        "id": "TC10",
        "category": "Guardrail / Out-of-Domain",
        "source_file": "None (Out of Domain)",
        "query": "What is the warranty coverage duration for Aventro electric motorbikes?",
        "ground_truth_context": (
            "N/A (The corpus only contains information about Aventro cars and SUVs, no motorbikes)."
        ),
        "ground_truth_answer": (
            "I don't know. The provided documents do not contain information about electric motorbikes or their warranty coverage."
        ),
        "expected_keywords": ["don't know", "not in the context"]
    }
]


def save_golden_set(output_dir: str = ".") -> tuple[str, str]:
    json_path = os.path.join(output_dir, "golden_set.json")
    csv_path = os.path.join(output_dir, "golden_set.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(GOLDEN_SET, f, indent=2, ensure_ascii=False)

    keys = ["id", "category", "source_file", "query", "ground_truth_answer", "expected_keywords"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(GOLDEN_SET)

    print(f"[OK] Generated Golden Set ({len(GOLDEN_SET)} test cases)")
    print(f"     JSON: {json_path}")
    print(f"     CSV:  {csv_path}")
    return json_path, csv_path


if __name__ == "__main__":
    save_golden_set()

