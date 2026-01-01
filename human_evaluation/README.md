# Human Evaluation Study

This directory contains materials for the human evaluation study conducted in the EMPRA paper.

## Directory Structure

```
human_evaluation/
├── guideline/                    # Annotation guidelines and training materials
│   ├── human_annotation_guideline.md
│   └── calibration_quiz.md
├── annotations/                  # Human annotator labels
│   ├── human_annotations.csv
│   └── README.md
├── results/                      # Analysis results and statistics
│   └── annotation_analysis_by_method.csv
└── analyze_annotations_by_method.py  # Analysis script
```

## Components

### Guideline Directory

Contains materials for training and calibrating human annotators:

- **`human_annotation_guideline.md`** - Complete annotation instructions including:
  - Task objectives (Fluency and Imperceptibility assessment)
  - Evaluation criteria and rating scales (1-5 for fluency, Normal/Attacked for imperceptibility)
  - Examples and edge cases
  - Important notes about avoiding bias and using human judgment only

- **`calibration_quiz.md`** - Training quiz with practice items and reference answers to ensure annotator consistency before the main annotation task

### Annotations Directory

Contains the human annotation data:

- **`human_annotations.csv`** - Main annotation file containing all annotator responses
  - Format: Two-row header structure with document IDs, methods, full document text, and annotator ratings
  - Each document is evaluated by multiple annotators
  - Columns include: `id`, `doc_id`, `method`, `document`, and annotator-specific fluency and imperceptibility ratings
  - See `annotations/README.md` for detailed format specification

- **`README.md`** - Detailed documentation of the annotation file format and structure

### Results Directory

Contains aggregated analysis results:

- **`annotation_analysis_by_method.csv`** - Statistical analysis results by attack method
  - Columns: `method`, `num_documents`, `avg_imperceptibility`, `kappa_imperceptibility`, `avg_fluency`, `kappa_fluency_weighted`
  - Metrics calculated:
    - Average imperceptibility score (0=Attacked, 1=Normal)
    - Cohen's kappa for imperceptibility agreement between annotators
    - Average fluency score (1-5 scale)
    - Weighted Cohen's kappa for fluency agreement between annotators

### Analysis Script

- **`analyze_annotations_by_method.py`** - Python script to analyze annotations by attack method
  - Reads `annotations/human_annotations.csv`
  - Calculates statistics for each method (EMPRA, PRADA, IDEM, Original, etc.)
  - Outputs results to `results/annotation_analysis_by_method.csv`
  - Usage: `python analyze_annotations_by_method.py`

## Reproducibility

To reproduce the human evaluation study:

1. **Review Guidelines**: Read `guideline/human_annotation_guideline.md` for complete annotation instructions
2. **Train Annotators**: Use `guideline/calibration_quiz.md` to calibrate annotators before the main task
3. **Examine Annotations**: Review `annotations/human_annotations.csv` for all annotator responses
4. **Run Analysis**: Execute `analyze_annotations_by_method.py` to generate statistical results
5. **Review Results**: Check `results/annotation_analysis_by_method.csv` for aggregated metrics by method

## Evaluation Metrics

The study evaluates two main dimensions:

1. **Fluency** (1-5 scale):
   - 5 = Excellent (very clear, natural, grammatically correct)
   - 4 = Good (generally clear with minor issues)
   - 3 = Moderate (understandable but noticeably uneven)
   - 2 = Low (frequent issues reducing readability)
   - 1 = Poor (very difficult to understand)

2. **Imperceptibility** (Binary):
   - Normal = Document appears natural and free from manipulation
   - Attacked = Document contains noticeable signs of manipulation

Inter-annotator agreement is measured using:
- Cohen's kappa for imperceptibility (binary classification)
- Weighted Cohen's kappa for fluency (ordinal scale)

