# Annotation Files

This directory contains human annotator labels for the EMPRA evaluation study.

## Main Annotation File

**File Name:** `human_annotations.csv`

This file contains all annotations from all annotators in a single CSV file.

## CSV Format

The annotation file follows this structure:

### Header Row 1 (Column Names)
```
id,doc_id,method,document,annotator_01,annotator_01,annotator_02,annotator_02,...
```

### Header Row 2 (Column Types/Labels)
```
,,,,fluency,imperceptibility,fluency,imperceptibility,...
```

### Data Rows
Each row contains:
- `id`: Sequential identifier (0, 1, 2, ...)
- `doc_id`: Document ID from the collection
- `method`: Attack method (Original, EMPRA, PRADA, IDEM, AttChain, Brittle-BERT, etc.)
- `document`: Full text of the document being evaluated
- For each annotator, two columns:
  - `annotator_{ID}` (fluency): Fluency rating with label, e.g., "5 (Excellent)", "4 (Good)", "3 (Moderate)", "2 (Low)", "1 (Poor)"
  - `annotator_{ID}` (imperceptibility): Binary label "Normal" or "Attacked"

### Column Structure
| Column | Description | Example |
|--------|-------------|---------|
| `id` | Sequential row ID | 0, 1, 2, ... |
| `doc_id` | Document identifier | 1088880, 8123101, ... |
| `method` | Attack method name | Original, EMPRA, AttChain, Brittle-BERT |
| `document` | Full document text | "Using the oxidation numbers..." |
| `annotator_01` (fluency) | Annotator 1 fluency rating | "5 (Excellent)" |
| `annotator_01` (imperceptibility) | Annotator 1 imperceptibility label | "Normal" or "Attacked" |
| `annotator_02` (fluency) | Annotator 2 fluency rating | "4 (Good)" |
| `annotator_02` (imperceptibility) | Annotator 2 imperceptibility label | "Normal" or "Attacked" |
