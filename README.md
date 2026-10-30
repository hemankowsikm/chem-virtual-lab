# 🧪 Cheminformatics Virtual Laboratory — v2.0

**SRM Institute of Science and Technology, Tiruchirappalli**
Division of Chemistry · School of Sciences · Faculty of Engineering and Technology

## Aim
To develop a foundational understanding of cheminformatics by using computational tools to
represent, visualise and analyse chemical structures.

## Learning Objectives
Students will be able to:

- Write, decode and validate **SMILES**; understand canonical SMILES, InChI and InChIKey.
- Visualise molecules in **2D and interactive 3D**, and measure bond lengths and angles.
- Calculate **20+ molecular descriptors** (MW, LogP, TPSA, HBD, HBA, rotatable bonds, rings, σ/π bonds, Fsp3, QED …).
- Apply **Lipinski, Veber and Egan** drug-likeness rules and interpret the results.
- Compare molecules with charts, **fingerprint (Tanimoto) similarity** and **SMARTS substructure search**.
- Understand structure–property relationships.

## What's new in v2.0
| Area | Upgrade |
|---|---|
| Navigation | 9 modules, SRM branding, student progress tracker |
| SMILES | New **SMILES Playground**: atom/bond tables, InChI, naming challenges |
| 3D | Atom labels, auto-rotate, vdW surface, force-field energy, MOL download, **geometry explorer** |
| Properties | QED, stereocentres, formal charge, **drug-likeness rules + radar plot**, functional-group detection with highlighting |
| Compare | CSV upload, bar charts, TPSA–LogP permeability map, structure gallery |
| Similarity | Morgan / RDKit / MACCS fingerprints, Tanimoto heat-map, ranking, SMARTS search |
| Assessment | Random 10 of 20 questions, answer explanations, best-score tracking, **completion certificate (PDF)** |
| Reports | Branded multi-molecule PDF with 2D structures and interpretation |
| Performance | Cached descriptor and 3D calculations |

## Files
```
app.py              main Streamlit application
requirements.txt    Python packages
packages.txt        system packages (Streamlit Community Cloud)
srm_logo.jpeg       institutional logo (keep next to app.py)
.streamlit/config.toml   SRM blue theme
```

## Installation
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Community Cloud
Upload all files (including the `.streamlit` folder and `srm_logo.jpeg`) to one GitHub repository
and set `app.py` as the main file.

## CSV format for *Compare Molecules*
```csv
name,smiles
Aspirin,CC(=O)Oc1ccccc1C(=O)O
Caffeine,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

## Developed by
Heman Kowsik M - Cybersecurity A
