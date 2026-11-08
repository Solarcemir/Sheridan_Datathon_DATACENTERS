#!/usr/bin/env python3
"""Merge offline-written investigations into output/watson_miss_explanations.json without any API call.

Reads offline_investigations.json (written by the Cursor agent from the same evidence packets that
watson_explain_misses.py sends to watsonx.ai), scores every entry with the same check_answer()
validator, and records its source. Investigations that already passed as IBM watsonx.ai output are
never replaced; the script verifies their content is byte-identical after the merge.

    Amy/feature/.venv/Scripts/python Amy/feature/apply_offline_investigations.py
    Amy/feature/.venv/Scripts/python Amy/feature/watson_explain_misses.py --recheck
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import watson_explain_misses as w  # noqa: E402

OFFLINE_PATH = w.HERE / "offline_investigations.json"
ANSWER_KEYS = ("what_happened", "possible_explanations", "analyst_lessons", "caveats")
WATSONX = "ibm_watsonx_ai"
LABELS = {
    "cursor_generated": ("Cursor-generated hypotheses and lessons, written offline from the same evidence packet and "
                         "checked by the same validator; not produced by IBM watsonx.ai. Not verified causes; analyst "
                         "review required."),
    "cursor_revised_watsonx_draft": ("IBM watsonx.ai draft revised offline by the Cursor agent to fix validator "
                                     "problems; the unmodified draft is kept in watsonx_original. Not verified "
                                     "causes; analyst review required."),
}


def fingerprint(ai: dict) -> str:
    return json.dumps({k: ai.get(k) for k in ANSWER_KEYS + ("model_id", "generated_at_utc", "label")},
                      sort_keys=True, ensure_ascii=False)


def main() -> int:
    evidence = w.load_evidence()
    cases = {c["cve_id"]: c for c in evidence["cases"]}
    results = json.loads(w.RESULT_PATH.read_text(encoding="utf-8"))["investigations"]
    offline = json.loads(OFFLINE_PATH.read_text(encoding="utf-8"))

    protected = {k: fingerprint(v["ai_generated"]) for k, v in results.items()
                 if v["ai_generated"]["status"] == "passed_checks"
                 and v["ai_generated"].get("source", WATSONX) == WATSONX}
    for task_id, entry in offline["investigations"].items():
        if task_id in protected:
            raise ValueError(f"{task_id} already passed as watsonx.ai output; refusing to replace it")
        if task_id not in cases or cases[task_id]["classification"] != "missed":
            raise ValueError(f"{task_id} is not one of the 43 missed KEV positives")
        source = entry["source"]
        if source not in LABELS:
            raise ValueError(f"{task_id}: unknown source {source!r}")

        packet = w.case_packet(cases[task_id], evidence)
        answer = {k: entry[k] for k in ANSWER_KEYS}
        problems = w.check_answer(answer, packet, "case")
        previous = results.get(task_id, {}).get("ai_generated", {})
        ai = {"status": "passed_checks" if not problems else "needs_review", "problems": problems,
              "attempts": None, **answer, "source": source, "model_id": offline["author"],
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "label": LABELS[source]}
        if source == "cursor_revised_watsonx_draft":
            original = previous.get("watsonx_original") or (previous if previous.get("source", WATSONX) == WATSONX else None)
            if not original or not original.get("what_happened"):
                raise ValueError(f"{task_id}: marked as a revision but no watsonx.ai draft exists")
            ai["revision_notes"] = entry["revision_notes"]
            ai["watsonx_original"] = original
        results[task_id] = {"kind": "case", "cve_id": task_id, "classification": "missed",
                            "verified_findings": packet["verified_findings"], "ai_generated": ai}
        print(f"{task_id}: {source} -> {ai['status']}" + (f": {problems}" if problems else ""))

    for record in results.values():
        record["ai_generated"].setdefault("source", WATSONX)
    changed = [k for k, fp in protected.items() if fingerprint(results[k]["ai_generated"]) != fp]
    if changed:
        raise RuntimeError(f"Protected watsonx.ai investigations changed: {changed}")
    w.write_results(results)
    print(f"Merged {len(offline['investigations'])} offline investigations; {len(protected)} watsonx.ai "
          f"investigations verified unchanged. Saved to {w.RESULT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
