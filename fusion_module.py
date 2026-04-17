"""
Decision fusion logic for smart mine entry authorization.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


PPE_ITEMS = ("helmet", "vest", "shoes")


def calculate_compliance_score(detection_result: Dict[str, object]) -> float:
    """Return the fraction of required PPE items detected as present."""
    present_count = sum(detection_result[item]["status"] == "present" for item in PPE_ITEMS)
    return round(present_count / len(PPE_ITEMS), 2)


def make_decision(detection_result: Dict[str, object]) -> str:
    """Convert PPE states into the final authorization decision."""
    if detection_result.get("reframe_required", False):
        return "REFRAME"
    if all(detection_result[item]["status"] == "present" for item in PPE_ITEMS):
        return "ALLOW"
    return "DENY"


def decision_color(decision: str) -> Tuple[int, int, int]:
    """Return the display color for the final decision."""
    if decision == "ALLOW":
        return (0, 200, 0)
    if decision == "DENY":
        return (0, 0, 255)
    return (0, 215, 255)


def build_authorization_result(detection_result: Dict[str, object]) -> Dict[str, object]:
    """Build the normalized authorization payload used by CLI, UI, and webcam."""
    decision = make_decision(detection_result)
    score = calculate_compliance_score(detection_result)

    missing_items = [
        item.title() if item != "shoes" else "Shoes"
        for item in PPE_ITEMS
        if detection_result[item]["status"] == "missing"
    ]
    uncertain_items = [
        item.title() if item != "shoes" else "Shoes"
        for item in PPE_ITEMS
        if detection_result[item]["status"] == "uncertain"
    ]

    if decision == "ALLOW":
        message = "All required PPE items detected."
    elif decision == "REFRAME":
        message = detection_result.get("reframe_message", "Adjust position to fit inside the frame.")
    else:
        message = "Missing required PPE: " + ", ".join(missing_items)

    return {
        "source": detection_result.get("source", ""),
        "decision": decision,
        "decision_color": decision_color(decision),
        "ppe_compliant": decision == "ALLOW",
        "compliance_score": score,
        "helmet_status": detection_result["helmet"]["status"],
        "vest_status": detection_result["vest"]["status"],
        "shoes_status": detection_result["shoes"]["status"],
        "helmet_confidence": detection_result["helmet"]["confidence"],
        "vest_confidence": detection_result["vest"]["confidence"],
        "shoes_confidence": detection_result["shoes"]["confidence"],
        "worker_count": detection_result.get("worker_count", 0),
        "person_detected": detection_result.get("person_detected", False),
        "missing_items": missing_items,
        "uncertain_items": uncertain_items,
        "message": message,
    }


def status_rows(authorization_result: Dict[str, object]) -> List[Dict[str, str]]:
    """Return simple row data for UI display."""
    return [
        {"PPE Item": "Helmet", "Status": authorization_result["helmet_status"].upper()},
        {"PPE Item": "Safety Vest", "Status": authorization_result["vest_status"].upper()},
        {"PPE Item": "Safety Shoes", "Status": authorization_result["shoes_status"].upper()},
        {"PPE Item": "Compliance Score", "Status": f"{authorization_result['compliance_score']:.2f}"},
        {"PPE Item": "Decision", "Status": authorization_result["decision"]},
    ]


def format_terminal_report(authorization_result: Dict[str, object]) -> str:
    """Format the final authorization result for terminal output."""
    return "\n".join(
        [
            "---------------------------------",
            f"Source: {authorization_result.get('source', 'N/A')}",
            f"Helmet: {authorization_result['helmet_status'].upper()}",
            f"Vest: {authorization_result['vest_status'].upper()}",
            f"Shoes: {authorization_result['shoes_status'].upper()}",
            f"Compliance Score: {authorization_result['compliance_score']:.2f}",
            f"FINAL DECISION: {authorization_result['decision']}",
            f"Guidance: {authorization_result['message']}",
            "---------------------------------",
        ]
    )
