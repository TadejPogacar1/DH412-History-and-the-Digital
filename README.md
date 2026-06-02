# DH412 History and the Digital

This repository contains a digital-history project on ballet technique at the Prix de Lausanne. The project focuses especially on the arabesque position and asks how ballet movement, training background, country/tradition, and competition history change across time.

The work combines manually cleaned competition metadata, candidate records, video-based pose extraction, and visual/statistical analysis of arabesque geometry.

## Project Structure

### `Data/`

Contains cleaned CSV files used for the arabesque analysis. These files include processed metadata and extracted movement measurements, such as arabesque angle, torso tilt, split angle, pelvic tilt, support-leg alignment, and shoulder-hip torsion.

Main files:

- `arabesque_analysis_ready_updated.csv`
- `arabesque_cleaned_for_analysis.csv`

### `Number of candidates and selected candidates in each year/`

Contains yearly PDFs and extracted candidate information for the Prix de Lausanne. These files are used to study the number of candidates, selected candidates, and competition participation patterns over time.

### `results/`

Stores outputs from the video and pose-analysis pipeline. It includes per-edition arabesque metric CSV files, extracted key frames, skeleton JSON files, and intermediate results generated from performance videos.

### `scripts/`

Contains reusable Python scripts for the movement-analysis pipeline:

- `extract_arabesque.py`: extracts arabesque pose metrics from videos.
- `extract_additional_features.py`: computes additional movement or pose features.
- `extract_frame_row.py`: extracts frame-level information.
- `merge_metadata.py`: merges movement results with metadata.

### `visualization/`

Contains generated figures and animations for the final analysis, including decade comparisons, PCA plots, trend plots, distribution plots, and male/female arabesque evolution GIFs.

### `scratch/`

Contains helper scripts used during development, such as notebook search and pipeline-update utilities. These are not the main analysis files.

### `xtcocotools/`

A lightweight local compatibility folder for COCO-style pose-evaluation imports used by the pose-estimation pipeline.

### `.cache/` and `.idea/`

Local environment/editor files. They are not part of the core research analysis.

## Main Notebooks

### `Data_Preprocessing_and_Analysis.ipynb`

Documents the metadata-cleaning process and general competition analysis. It explains manual corrections, gender labeling, missing years, classical/contemporary variation filtering, and the construction of analysis datasets.

It also analyzes finalists by country, gender, school, year, and awards.

### `pipeline_arabesque.ipynb`

Runs the arabesque extraction pipeline module by module. This notebook is mainly used to process videos and generate movement metrics.

### `arabesque_analysis.ipynb`

Analyzes the extracted arabesque measurements. It includes feature correlations, temporal trends, standardization/diversity analysis, generational changes, motion extremes, and 3D skeleton visualization.

### `arabesque_country_before2011.ipynb`

Compares arabesque geometry across country/tradition groups before 2011. It groups dancers into traditions such as French, Russian/Vaganova, British/RAD, Balanchine/American, and Others, then compares six geometry metrics.

### `arabesque_school_2012-2026.ipynb`

Analyzes school/style differences from 2012 to 2026. It compares school-style groups, decade patterns, geometry profiles, and whether time-period effects remain after controlling for training style.

## Environment

The project uses a Conda environment with Python 3.10 and packages for data analysis, visualization, video processing, and pose estimation.

```bash
conda env create -f environment.yml
conda activate ballet_env
```

Important packages include `pandas`, `numpy`, `matplotlib`, `opencv-python`, `moviepy`, `mmpose`, `mmdet`, and `scikit-learn`.

