# DH412 History and the Digital

This project studies the evolution of ballet technique in Prix de Lausanne performances, with a focus on the arabesque position. It combines competition metadata, selected-candidate records, video-based pose extraction, and statistical/visual analysis.

## Contents

- `Data/`: cleaned datasets for arabesque and competition analysis.
- `Data_Preprocessing_and_Analysis.ipynb`: metadata cleaning and general analysis of finalists, countries, gender, and awards.
- `pipeline_arabesque.ipynb`: pose-processing pipeline for extracting arabesque metrics from videos.
- `arabesque_analysis.ipynb`: correlation, temporal trend, diversity, and motion-extreme analysis.
- `arabesque_country_before2011.ipynb`: comparison of arabesque geometry across country/tradition groups before 2011.
- `arabesque_school_2012-2026.ipynb`: comparison of ballet school/style groups from 2012 to 2026.
- `results/`: extracted metrics, frames, skeleton data, and generated outputs.

## Setup

Create the environment with:

```bash
conda env create -f environment.yml
conda activate ballet_env
```

## Usage

Run the notebooks from the project root. Start with `Data_Preprocessing_and_Analysis.ipynb` for the dataset overview, then use the arabesque notebooks for movement and style analysis.

