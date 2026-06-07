# Curriculum Patterns Report

Analysis of DSEP / El Camino notebook corpus for curriculum design patterns.

*Generated from 10 notebooks in `data/parsed/`*

---

## 1. Notebook Overview

**Total notebooks analyzed:** 10

### Disciplines represented

- Ethics / Data Science: 1 (10%)
- Environmental Science: 1 (10%)
- Social Science / Criminal Justice: 1 (10%)
- Economics / Housing: 1 (10%)
- Chemistry / Physics: 1 (10%)
- Biology / Genomics: 1 (10%)
- Biology / Biomanufacturing: 1 (10%)
- Business / Marketing: 1 (10%)
- Public Health / Epidemiology: 1 (10%)
- Mathematics / Calculus: 1 (10%)

### Coding level distribution

- Beginner: 4 (40%)
- Intermediate: 5 (50%)
- Advanced: 1 (10%)

### Most common libraries used

- matplotlib: 8 (80%)
- pandas: 7 (70%)
- numpy: 7 (70%)
- plotly: 5 (50%)
- ipywidgets: 5 (50%)
- seaborn: 4 (40%)

---

## 2. Teaching Pattern Analysis

### Most common notebook structures

- **Introduction → Interactive Widget → Analysis → Reflection** — 1 notebook(s)
- **Learning Objectives → Introduction** — 1 notebook(s)
- **Introduction → Sources → Exercise → Reflection** — 1 notebook(s)
- **Analysis → Introduction → Dataset** — 1 notebook(s)
- **Visualization → Background → Conclusion** — 1 notebook(s)
- **Dataset** — 1 notebook(s)
- **Analysis → Learning Objectives → Introduction → Dataset → Visualization → Reflection** — 1 notebook(s)
- **Introduction** — 1 notebook(s)
- **Background → Visualization** — 1 notebook(s)
- **Introduction → Learning Objectives → Exercise → Visualization → Conclusion → Reflection** — 1 notebook(s)

### Most common section orders

- Introduction → Interactive Widget → Analysis → Reflection (1)
- Learning Objectives → Introduction (1)
- Introduction → Sources → Exercise → Reflection (1)
- Analysis → Introduction → Dataset (1)
- Visualization → Background → Conclusion (1)
- Dataset (1)
- Analysis → Learning Objectives → Introduction → Dataset → Visualization → Reflection (1)
- Introduction (1)
- Background → Visualization (1)
- Introduction → Learning Objectives → Exercise → Visualization → Conclusion → Reflection (1)

### Most common learning activity types

- Exercises / Questions: 7 (70%)
- Real-World Context: 7 (70%)
- Answers / Solutions: 6 (60%)
- Reflection Questions: 4 (40%)
- Hints: 3 (30%)

### Most common visualization activities

- matplotlib pyplot calls: 7 notebook(s)
- pandas/matplotlib .plot(): 4 notebook(s)
- seaborn calls: 3 notebook(s)
- scatter plot: 3 notebook(s)
- subplots: 3 notebook(s)
- plotly express: 3 notebook(s)
- heatmap: 2 notebook(s)
- bar chart: 2 notebook(s)
- plotly graph objects: 1 notebook(s)
- histogram: 1 notebook(s)

### Most common reflection / question styles

- Reflection Prompts: 6 notebook(s)
- Answer Placeholder Cells: 6 notebook(s)
- Numbered Questions: 6 notebook(s)
- Discussion Prompts: 5 notebook(s)
- Scenario-Based Prompts: 1 notebook(s)
- External Reflection Form: 1 notebook(s)
- Free Response Questions: 1 notebook(s)

---

## 3. Interactivity Analysis

**Notebooks using widgets:** 6 of 10

- AI_Ethics_in_Data_Science.ipynb
- Incarcerations.ipynb
- Orbitals.ipynb
- marketing_case_study.ipynb
- parental-risk.ipynb
- sums.ipynb

### Types of widgets used

- ipywidgets library: 5 notebook(s)
- widgets API: 5 notebook(s)
- @interact decorator / interact(): 3 notebook(s)
- dropdown widget: 3 notebook(s)
- external widget file (%run): 1 notebook(s)
- custom ethics widget: 1 notebook(s)
- embedded YouTube video: 1 notebook(s)
- button widget: 1 notebook(s)
- slider widget: 1 notebook(s)

### How widgets are incorporated into learning

- **AI_Ethics_in_Data_Science.ipynb**: External widget file keeps notebook readable
- **Incarcerations.ipynb**: Embedded video for follow-along guidance
- **Orbitals.ipynb**: `@interact` / `interact()` for parameter exploration
- **marketing_case_study.ipynb**: Interactive elements support exploration sections
- **parental-risk.ipynb**: `@interact` / `interact()` for parameter exploration
- **sums.ipynb**: `@interact` / `interact()` for parameter exploration

---

## 4. Notebook Archetypes

### Introductory EDA Notebook

**Defining characteristics:** Guides students through exploring a dataset with tables, summaries, and charts.

**Common section order:** Introduction → Learning Objectives → Dataset → Exploration → Visualization

**Common teaching methods:**
- Load and preview data early
- Column guides and data dictionaries
- Guided plots with interpretation prompts

**Representative notebooks:**
- Incarcerations.ipynb
- cancer_mutations.ipynb
- enzyme.ipynb
- marketing_case_study.ipynb
- parental-risk.ipynb

### Coding-Focused Notebook

**Defining characteristics:** Students write and test code, often with autograding, hints, and scaffolded functions.

**Common section order:** Introduction → Dataset → Coding Exercises → Model Building → Evaluation

**Common teaching methods:**
- Step-by-step coding tasks with blanks to fill in
- Hints before harder implementation steps
- Otter or inline answer-checking

**Representative notebooks:**
- Incarcerations.ipynb
- Modeling_CA_Housing_lab.ipynb
- cancer_mutations.ipynb
- enzyme.ipynb
- marketing_case_study.ipynb

### Widget-Based Interactive Notebook

**Defining characteristics:** Uses sliders, dropdowns, or custom widgets so students explore concepts interactively.

**Common section order:** Background → Interactive Widget → Experimentation → Discussion

**Common teaching methods:**
- Run-once setup cell for widget imports
- Parameter exploration without rewriting code
- External widget files for complex UIs

**Representative notebooks:**
- Incarcerations.ipynb
- Orbitals.ipynb
- marketing_case_study.ipynb
- parental-risk.ipynb
- sums.ipynb

### Reflection-Based Notebook

**Defining characteristics:** Prioritizes discussion, ethical reasoning, or written reflection over heavy coding.

**Common section order:** Introduction → Guided Exploration → Reflection → Conclusion

**Common teaching methods:**
- Free-response or short-answer prompts
- Scenario-based discussion cues
- Post-notebook reflection forms

**Representative notebooks:**
- AI_Ethics_in_Data_Science.ipynb
- Incarcerations.ipynb
- enzyme.ipynb
- sums.ipynb

### Case Study Notebook

**Defining characteristics:** Frames learning around a real-world scenario, policy issue, or industry problem.

**Common section order:** Case Introduction → Data / Context → Analysis → Application → Wrap-Up

**Common teaching methods:**
- Narrative framing with domain vocabulary
- Multi-section storyline (e.g. 4 P's, policy eras)
- Connect data patterns to real decisions

**Representative notebooks:**
- AI_Ethics_in_Data_Science.ipynb
- Incarcerations.ipynb
- Modeling_CA_Housing_lab.ipynb
- cancer_mutations.ipynb
- enzyme.ipynb
- marketing_case_study.ipynb
- parental-risk.ipynb

---

## 5. Reusable Templates

Template structures that appear reusable across the corpus:

### Template B: Interactive Exploration

Introduction
Interactive Widget
Analysis
Reflection

*Observed in 1 notebook(s)*

**Example notebooks:** AI_Ethics_in_Data_Science.ipynb

### Template: Learning Objectives-Focused

Learning Objectives
Introduction

*Observed in 1 notebook(s)*

**Example notebooks:** ECC CalEnviroScreen.ipynb

### Template C: Exercise + Reflection

Introduction
Sources
Exercise
Reflection

*Observed in 1 notebook(s)*

**Example notebooks:** Incarcerations.ipynb

### Template: Analysis + Introduction + Dataset

Analysis
Introduction
Dataset

*Observed in 1 notebook(s)*

**Example notebooks:** Modeling_CA_Housing_lab.ipynb

### Template: Visualization + Background + Conclusion

Visualization
Background
Conclusion

*Observed in 1 notebook(s)*

**Example notebooks:** Orbitals.ipynb

### Template: Dataset-Focused

Dataset

*Observed in 1 notebook(s)*

**Example notebooks:** cancer_mutations.ipynb

### Template A: Structured Data Lesson

Analysis
Learning Objectives
Introduction
Dataset
Visualization
Reflection

*Observed in 1 notebook(s)*

**Example notebooks:** enzyme.ipynb

### Template: Introduction-Focused

Introduction

*Observed in 1 notebook(s)*

**Example notebooks:** marketing_case_study.ipynb

### Template: Background-Focused

Background
Visualization

*Observed in 1 notebook(s)*

**Example notebooks:** parental-risk.ipynb

### Template A: Structured Data Lesson

Introduction
Learning Objectives
Exercise
Visualization
Conclusion
Reflection

*Observed in 1 notebook(s)*

**Example notebooks:** sums.ipynb

### Template A: Introductory Data Lesson

Introduction
Learning Objectives
Dataset Overview
Visualization
Interpretation Questions
Reflection


**Example notebooks:** Incarcerations.ipynb, enzyme.ipynb, sums.ipynb

### Template B: Interactive Exploration

Background
Interactive Widget
Experimentation
Discussion


**Example notebooks:** AI_Ethics_in_Data_Science.ipynb, Incarcerations.ipynb, Orbitals.ipynb, marketing_case_study.ipynb, parental-risk.ipynb, sums.ipynb

### Template C: Coding Lab with Scaffolding

Introduction
Dataset
Guided Coding
Hints
Model / Analysis
Evaluation


**Example notebooks:** Modeling_CA_Housing_lab.ipynb, cancer_mutations.ipynb, marketing_case_study.ipynb

---

## 6. Recommendations

Based on this notebook corpus:

- **Core libraries**: matplotlib, pandas, numpy, plotly, ipywidgets appear in most notebooks — include import/setup cells and brief library introductions in generated notebooks.
- **High-value activities**: Exercises / Questions, Real-World Context, Answers / Solutions, Reflection Questions, Hints are common — generated notebooks should plan for these components explicitly.
- **Reusable structure**: The most frequent detected pattern is "Introduction → Interactive Widget → Analysis → Reflection". Use this as a default outline when topics fit.
- **Archetypes to model**: Introductory EDA Notebook, Coding-Focused Notebook, Widget-Based Interactive Notebook, Reflection-Based Notebook, Case Study Notebook each appear multiple times — store archetype tags in metadata for retrieval.
- **Store for generation**: title, learning objectives, section order, dataset description, exercise prompts, hint placement, reflection style, visualization types, widget patterns, coding level, and discipline/topic.
- **Style consistency**: Many notebooks open with estimated time, table of contents, and a library-import cell — preserve these as optional template blocks.
