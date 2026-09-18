# Data Card: Brain Cancer Clinical Dataset

## Summary

`data/brain_cancer_clinical_data.csv` contains de-identified clinical and
molecular-pathology records for a cohort of brain tumor patients, used to
predict the binary outcome `Event_death` (0 = survived, 1 = deceased at last
follow-up).

- **Rows:** ~63 patients
- **Target:** `Event_death` (binary)
- **Identifiers:** the original `Patient` column is a study-internal sequence
  number only; no names, dates of birth, or other direct identifiers are
  present. It is dropped before modeling and is not linkable to any external
  registry from this repository alone.

## ⚠️ Ethical note

This is a **real, small, single-cohort clinical dataset**, not synthetic
data. Even without direct identifiers, clinical datasets of this kind can
carry re-identification risk when combined with other sources, and any
findings here reflect one cohort and must **not** be interpreted as
generalizable medical guidance. If you plan to reuse or redistribute this
data beyond this repository, or to apply this pipeline to a different
clinical cohort, please:

- confirm you have the right/authorization to do so under your institution's
  data governance and applicable regulations (e.g. HIPAA, GDPR, or local
  equivalents), and
- treat any model output as a research artifact, not a diagnostic or
  treatment decision tool.

## Schema

| Column | Type | Description |
| --- | --- | --- |
| `gender` | categorical | Patient sex code |
| `Age` | numeric | Age in years |
| `timefordeath` | numeric | Follow-up time to death or censoring |
| `time_Recurrence` | numeric | Time to tumor recurrence |
| `Event_reccurrence` | numeric (0/1) | Whether recurrence occurred |
| `PCV_new` | numeric | Chemotherapy regimen indicator |
| `TMZ` | categorical | Temozolomide treatment status |
| `Radiology` | categorical | Radiologic classification code |
| `DxWHO2007` | categorical | WHO 2007 diagnostic classification |
| `IDH1_molecular`, `IDH1_2`, `IDH1_tarkibi`, `IDH2` | categorical | IDH mutation status markers |
| `H3.3K27M`, `H3.3G34R`, `H3F3A` | categorical | Histone H3 mutation markers |
| `BRAFV600E`, `V600E` | categorical | BRAF V600E mutation markers |
| `EGFR`, `EGFR_A` | categorical | EGFR alteration markers |
| `MGMT`, `MGMT_new` | categorical | MGMT promoter methylation markers |
| `@1p19q` | categorical | 1p/19q co-deletion status |
| `PTEN` | categorical | PTEN alteration status |
| `CD44` | categorical | CD44 expression marker |
| `IntegratedDxStep1`, `Integrateddxstep2` | categorical | Integrated (histo-molecular) diagnosis codes |
| `Event_death` | target (0/1) | Vital status at last follow-up |

Categorical columns are stored as small integer codes in the source study's
own coding scheme; see the original study documentation for the exact
mapping of each code if available.

## Known data-quality artifacts (handled by this codebase)

- A small number of cells contain a single whitespace character instead of
  being truly empty; `brain_cancer_classifier.data_loading.load_dataset`
  normalizes these to proper missing values before imputation.
