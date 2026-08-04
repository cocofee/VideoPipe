# Cycling Event Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent runners and other non-cyclists from becoming official cycling events while preserving valid cyclists, event IDs, original-frame OCR evidence, and laptop usability.

**Architecture:** Add an explicit cycling sport profile and apply the existing lightweight bicycle validator to both bibbed and unbibbed pending events. Validate the athlete crop together with expanded original-frame context, reject only explicit negative verdicts, expose rejected candidates to offline regression reports, and leave tracker-handoff deduplication as the next separately verified change after the admission baseline is stable.

**Tech Stack:** Python, numpy, OpenCV, Ultralytics YOLO, pytest.

---

### Task 1: Cycling profile boundary

**Files:**
- Modify: `realtime/detector.py`
- Test: `tests/realtime/test_detection_association.py`

- [ ] **Step 1: Write the failing profile tests**

Add tests asserting that the default profile is `cycling`, `sport_profile="cycling"` is accepted, and unsupported profiles such as `running` raise `ValueError` in this phase.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/realtime/test_detection_association.py -k "sport_profile" -q
```

Expected: FAIL because `Detector` does not yet expose or validate `sport_profile`.

- [ ] **Step 3: Implement the minimal profile boundary**

Add a `sport_profile: str = "cycling"` constructor argument, normalize it once, store `self.sport_profile`, and raise a clear `ValueError` for profiles not implemented in this phase.

- [ ] **Step 4: Verify GREEN**

Run the same focused command and expect all selected tests to pass.

### Task 2: Validate bibbed cycling candidates

**Files:**
- Modify: `realtime/detector.py`
- Modify: `tests/realtime/test_detection_association.py`

- [ ] **Step 1: Replace the obsolete bypass test with failing cycling tests**

Replace `test_bib_evidence_skips_secondary_athlete_validation` with tests proving:

- a bibbed cycling candidate is rejected when the validator returns only `person`;
- a bibbed cyclist is accepted when both athlete and context evidence include `bicycle`;
- a bicycle seen only in the tight crop or only in background context is not sufficient;
- missing validators and validator exceptions fail open.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest tests/realtime/test_detection_association.py -k "validator or bib_evidence" -q
```

Expected: the bibbed-candidate rejection and multi-evidence tests fail against the current bypass behavior.

- [ ] **Step 3: Implement multi-evidence event validation**

In `realtime/detector.py`:

- remove the `state.best_bib or state.has_bib_box` early return;
- create expanded context crops from `pending_event_data["frame"]` and the original bounding box;
- require bicycle evidence in the athlete crop and in at least one valid expanded context crop;
- preserve fail-open behavior when the model is absent or raises an exception;
- keep validation event-only and before event ID allocation.

- [ ] **Step 4: Verify GREEN and no event-ID regression**

Run:

```powershell
python -m pytest tests/realtime/test_detection_association.py -q
```

Expected: all detection-association tests pass, including the existing rejected-candidate event-ID test.

### Task 3: Rejected-candidate regression evidence

**Files:**
- Modify: `realtime/detector.py`
- Modify: `realtime/devtools/regression_report.py`
- Modify: `tests/realtime/test_regression_metrics.py`

- [ ] **Step 1: Write failing report tests**

Add tests proving that explicit cycling-validator rejections are counted separately from official events and do not increment crossing-event totals.

- [ ] **Step 2: Run the focused report tests and verify RED**

Run:

```powershell
python -m pytest tests/realtime/test_regression_metrics.py -q
```

Expected: FAIL because rejected candidates are not yet exposed to the report.

- [ ] **Step 3: Implement bounded rejected-candidate output**

Expose per-frame rejected-candidate records containing track ID, bbox, timestamp, reason, bib-box presence, original frame, athlete crop, and bib crop. Keep the list bounded and regression-only; do not write RaceData or allocate event IDs. Extend the offline report to count records and optionally save their evidence under the requested evidence directory.

- [ ] **Step 4: Verify GREEN**

Run both focused test files and expect all tests to pass.

### Task 4: Cycling video regression

**Files:**
- Generate only: `runs/repository_audit_20260804/cycling_admission_matrix/**`

- [ ] **Step 1: Run the marathon negative regression**

Use the deployed cycling model plus `yolo11n.pt` on all 8,979 frames. Expected: zero official cycling events, with rejected candidates retained in the report.

- [ ] **Step 2: Run positive short videos**

Run `3.mp4`, `4.mp4`, and the `408...mp4` city finish. Expected: retain the previously verified 2, 5, and 7 official events unless evidence inspection proves an existing event is invalid.

- [ ] **Step 3: Run mountain source and HEVC variants**

Record remaining duplicate counts without changing deduplication in the same patch. This isolates event admission from tracker-handoff behavior.

- [ ] **Step 4: Run repository verification**

Run:

```powershell
python -m pytest tests/realtime/test_detection_association.py tests/realtime/test_regression_metrics.py tests/realtime/test_ocr_duplicate_merge.py tests/realtime/test_ocr_init.py -q
python -m compileall realtime
git diff --check
```

Expected: tests pass, compilation succeeds, and no whitespace errors are reported.

No commit or push is performed without explicit user authorization.
