# CyberNER: A Harmonized STIX Corpus for Cybersecurity Named Entity Recognition

**Description:**

This repository contains the CyberNER project, focused on addressing schema heterogeneity in cybersecurity Named Entity Recognition (NER). We introduce **CyberNER**, a large-scale, unified corpus created by systematically harmonizing four prominent datasets (CyNER, DNRTI, APTNER, and Attacker) onto a consistent taxonomy based on the STIX 2.1 standard.

The primary goal of CyberNER is to overcome the challenges posed by incompatible annotation schemas in existing resources. By providing a standardized, STIX-aligned benchmark dataset, this project aims to facilitate the development, rigorous evaluation, and comparison of more robust, generalizable, and interoperable NER models for the cybersecurity domain.

This repository includes:
*   The harmonized CyberNER dataset.
*   Scripts used for data cleaning and schema harmonization.
*   Code for training and evaluating benchmark NER models (e.g., BERT-CRF variants) on the CyberNER corpus.