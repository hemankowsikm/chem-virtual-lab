"""
Cheminformatics Virtual Laboratory  (v2.0)
SRM Institute of Science and Technology, Tiruchirappalli
Division of Chemistry, School of Sciences

Run:  streamlit run app.py
"""

import io
import os
import random
import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors,
    rdMolTransforms, rdFingerprintGenerator, MACCSkeys, QED, Draw,
)
from rdkit.Chem.Draw import rdMolDraw2D

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

RDLogger.DisableLog("rdApp.*")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(BASE_DIR, "srm_logo.jpeg")
SRM_BLUE = "#14438f"
SRM_GOLD = "#c9a40e"

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide",
)

st.markdown(
    f"""
<style>
.hero {{
    background: linear-gradient(120deg, {SRM_BLUE} 0%, #1f6fd1 100%);
    color: white; padding: 22px 28px; border-radius: 14px; margin-bottom: 14px;
}}
.hero h1 {{ color: white; margin: 0; font-size: 2.1rem; }}
.hero p  {{ margin: 4px 0 0 0; opacity: .92; font-size: 1.02rem; }}
div[data-testid="stMetric"] {{
    background: #eef3fb; border-left: 5px solid {SRM_BLUE};
    padding: 8px 12px; border-radius: 8px;
}}
.pill-ok  {{ background:#dff5e3; color:#14692a; padding:2px 10px; border-radius:12px; font-weight:600; }}
.pill-bad {{ background:#fde2e2; color:#a11d1d; padding:2px 10px; border-radius:12px; font-weight:600; }}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# DATA
# ============================================================

EXAMPLES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Dopamine": "NCCc1ccc(O)c(O)c1",
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    "Glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}

DEFAULT_LIBRARY = "\n".join(
    f"{n},{s}" for n, s in list(EXAMPLES.items())[:8]
)

FUNCTIONAL_GROUPS = {
    "Alcohol (–OH)": "[CX4][OX2H]",
    "Phenol": "c[OX2H]",
    "Carboxylic acid": "[CX3](=O)[OX2H1]",
    "Ester": "[#6][CX3](=O)[OX2][#6]",
    "Ether": "[OD2]([#6])[#6]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Amide": "[CX3](=O)[NX3]",
    "Primary amine": "[NX3;H2][CX4]",
    "Secondary amine": "[NX3;H1]([CX4])[CX4]",
    "Tertiary amine": "[NX3;H0]([CX4])([CX4])[CX4]",
    "Nitro": "[N+](=O)[O-]",
    "Nitrile": "C#N",
    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "C#C",
    "Halide": "[F,Cl,Br,I]",
    "Thiol": "[SX2H]",
    "Sulfonamide": "S(=O)(=O)[NX3]",
    "Aromatic ring": "a1aaaaa1",
}
FG_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FUNCTIONAL_GROUPS.items()}

CHALLENGES = {
    "Methanol": "CO",
    "Ethanol": "CCO",
    "Acetone": "CC(C)=O",
    "Acetic acid": "CC(=O)O",
    "Benzene": "c1ccccc1",
    "Toluene": "Cc1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Ethylamine": "CCN",
    "Cyclohexane": "C1CCCCC1",
    "Acetonitrile": "CC#N",
}

# ============================================================
# STATE
# ============================================================

PAGES = [
    "🏠 Home",
    "📖 Theory",
    "🔤 SMILES Playground",
    "🧬 2D & 3D Visualization",
    "📊 Properties & Drug-likeness",
    "📈 Compare Molecules",
    "🔍 Similarity & Substructure",
    "📝 Assessment",
    "📄 Final Report",
]

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CC(=O)Oc1ccccc1C(=O)O"
if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Aspirin"
if "progress" not in st.session_state:
    st.session_state.progress = {"visited": set(), "molecules": set(), "best": None}


def load_example():
    st.session_state.active_smiles = EXAMPLES[st.session_state.selected_example]


# ============================================================
# CHEMISTRY HELPERS
# ============================================================

def bond_information(mol):
    """σ and π bond counts (σ includes bonds to implicit hydrogens)."""
    sigma = Chem.AddHs(mol).GetNumBonds()
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        pass
    extra = {Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.QUADRUPLE: 3}
    pi = 0.0
    for b in kek.GetBonds():
        pi += extra.get(b.GetBondType(), 0)
        if b.GetBondType() == Chem.BondType.AROMATIC:
            pi += 0.5
    return sigma, int(round(pi))


@st.cache_data(show_spinner=False)
def props_for(smiles):
    """All descriptors for a SMILES string (None if invalid)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    sigma, pi = bond_information(mol)
    try:
        qed = round(QED.qed(mol), 3)
    except Exception:
        qed = None
    try:
        inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        inchi_key = ""
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 3),
        "Exact Mass": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Total Atoms (with H)": Chem.AddHs(mol).GetNumAtoms(),
        "Heavy-Atom Bonds": mol.GetNumBonds(),
        "Sigma (σ) Bonds": sigma,
        "Pi (π) Bonds": pi,
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar Refractivity": round(Crippen.MolMR(mol), 2),
        "Formal Charge": Chem.GetFormalCharge(mol),
        "Stereocentres": len(centers),
        "QED (drug-likeness score)": qed,
        "Canonical SMILES": Chem.MolToSmiles(mol),
        "InChIKey": inchi_key,
    }


def drug_likeness(p):
    """Returns {rule name: [(criterion, value, passed), ...]}."""
    return {
        "Lipinski Rule of Five": [
            ("MW ≤ 500", p["Molecular Weight"], p["Molecular Weight"] <= 500),
            ("LogP ≤ 5", p["LogP"], p["LogP"] <= 5),
            ("HBD ≤ 5", p["HBD"], p["HBD"] <= 5),
            ("HBA ≤ 10", p["HBA"], p["HBA"] <= 10),
        ],
        "Veber (oral bioavailability)": [
            ("Rotatable bonds ≤ 10", p["Rotatable Bonds"], p["Rotatable Bonds"] <= 10),
            ("TPSA ≤ 140 Å²", p["TPSA"], p["TPSA"] <= 140),
        ],
        "Egan (absorption)": [
            ("TPSA ≤ 131.6 Å²", p["TPSA"], p["TPSA"] <= 131.6),
            ("LogP ≤ 5.88", p["LogP"], p["LogP"] <= 5.88),
        ],
    }


def lipinski_violations(p):
    return sum(1 for _, _, ok in drug_likeness(p)["Lipinski Rule of Five"] if not ok)


def functional_groups(mol):
    found = {}
    for name, patt in FG_PATTERNS.items():
        n = len(mol.GetSubstructMatches(patt))
        if n:
            found[name] = n
    return found


def interpret(p):
    notes = []
    mw, logp, tpsa = p["Molecular Weight"], p["LogP"], p["TPSA"]
    notes.append(
        f"**Size:** MW {mw} g/mol — "
        + ("small molecule, easily absorbed." if mw < 300
           else "typical drug-like size." if mw <= 500
           else "large; may have poor oral absorption.")
    )
    notes.append(
        f"**Lipophilicity:** LogP {logp} — "
        + ("hydrophilic, dissolves well in water." if logp < 0
           else "balanced water/lipid behaviour." if logp <= 3
           else "lipophilic; good membrane crossing, lower solubility." if logp <= 5
           else "very lipophilic; solubility and toxicity risks.")
    )
    notes.append(
        f"**Polarity:** TPSA {tpsa} Å² — "
        + ("low polarity, can cross membranes and often the blood–brain barrier." if tpsa < 60
           else "moderate polarity, good intestinal absorption." if tpsa <= 140
           else "highly polar, poor passive permeability.")
    )
    notes.append(
        f"**Flexibility:** {p['Rotatable Bonds']} rotatable bond(s) — "
        + ("rigid structure." if p["Rotatable Bonds"] <= 3
           else "moderately flexible." if p["Rotatable Bonds"] <= 10
           else "very flexible; entropy penalty on binding.")
    )
    notes.append(
        f"**Saturation:** Fsp3 = {p['Fraction Csp3']} — "
        + ("mostly flat / aromatic." if p["Fraction Csp3"] < 0.25 else "good three-dimensional character.")
    )
    return notes


def draw_mol(mol, size=(650, 440), highlight=None, indices=False):
    """PNG bytes of a 2D depiction."""
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    if indices:
        d.drawOptions().addAtomIndices = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=list(highlight) if highlight else []
    )
    d.FinishDrawing()
    return d.GetDrawingText()


@st.cache_data(show_spinner=False)
def build_3d(smiles):
    """Embed + optimise. Returns (molblock, energy, forcefield) or None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    energy, ff = None, "none"
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            energy = AllChem.MMFFGetMoleculeForceField(mol, mp).CalcEnergy()
            ff = "MMFF94"
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            energy = AllChem.UFFGetMoleculeForceField(mol).CalcEnergy()
            ff = "UFF"
    except Exception:
        pass
    return Chem.MolToMolBlock(mol), energy, ff


def show_3d(molblock, style, surface=False, labels=False, spin=False, height=520):
    view = py3Dmol.view(width=850, height=height)
    view.addModel(molblock, "mol")
    styles = {
        "Ball and Stick": {"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}},
        "Stick": {"stick": {"radius": 0.2}},
        "Space Filling": {"sphere": {"scale": 1.0}},
        "Wireframe": {"line": {"linewidth": 2}},
    }
    view.setStyle(styles[style])
    if surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.65, "color": "white"})
    if labels:
        m = Chem.MolFromMolBlock(molblock, removeHs=False)
        conf = m.GetConformer()
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue
            pos = conf.GetAtomPosition(a.GetIdx())
            view.addLabel(
                f"{a.GetSymbol()}{a.GetIdx()}",
                {"position": {"x": pos.x, "y": pos.y, "z": pos.z},
                 "fontSize": 13, "fontColor": "black",
                 "backgroundOpacity": 0.35, "showBackground": True},
            )
    view.setBackgroundColor("white")
    if spin:
        view.spin(True)
    view.zoomTo()
    components.html(view._make_html(), height=height + 20)


def parse_library(text):
    """'Name,SMILES' or bare SMILES per line → (records, invalid_lines)."""
    recs, bad = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            name, smi = [x.strip() for x in line.split(",", 1)]
        else:
            name, smi = f"Molecule {len(recs) + 1}", line
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(line)
        else:
            recs.append({"name": name or smi, "smiles": smi, "mol": mol})
    return recs, bad


def molecule_input():
    """Shared SMILES entry used by every single-molecule page."""
    c1, c2 = st.columns([2, 1])
    with c1:
        st.text_input("Enter SMILES", key="active_smiles")
    with c2:
        st.selectbox("Example molecule", list(EXAMPLES), key="selected_example")
        st.button("Load example", on_click=load_example)
    smiles = st.session_state.active_smiles.strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        st.error("❌ Invalid SMILES. Check brackets, ring-closure digits and atom symbols.")
        return None, smiles
    st.success(f"✅ Valid molecule — {rdMolDescriptors.CalcMolFormula(mol)}")
    st.session_state.progress["molecules"].add(Chem.MolToSmiles(mol))
    return mol, smiles


# ============================================================
# PDF HELPERS
# ============================================================

def build_report_pdf(student, reg, records):
    """records: list of dicts with name, smiles, mol."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36,
        title="Cheminformatics Virtual Laboratory Report",
    )
    ss = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=ss["Title"], textColor=HexColor(SRM_BLUE), fontSize=19)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=HexColor(SRM_BLUE))
    centre = ParagraphStyle("c", parent=ss["Normal"], alignment=TA_CENTER)
    small = ParagraphStyle("s", parent=ss["Normal"], fontSize=8.5, leading=11)
    body = ss["Normal"]

    story = []
    if os.path.exists(LOGO):
        story.append(RLImage(LOGO, width=3.2 * inch, height=1.6 * inch))
    story += [
        Paragraph("Cheminformatics Virtual Laboratory", title),
        Paragraph("Molecular Analysis Report", centre),
        Spacer(1, 10),
    ]
    info = Table(
        [
            ["Student Name", student or "-"],
            ["Registration No.", reg or "-"],
            ["Date", datetime.date.today().strftime("%d %B %Y")],
            ["Molecules analysed", str(len(records))],
        ],
        colWidths=[1.7 * inch, 4.5 * inch],
    )
    info.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), HexColor("#eef3fb")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    story += [info, Spacer(1, 12)]

    for i, rec in enumerate(records, 1):
        p = props_for(rec["smiles"])
        story.append(PageBreak() if i > 1 else Spacer(1, 4))
        story.append(Paragraph(f"{i}. {escape(rec['name'])}", h2))
        story.append(Paragraph(f"<b>SMILES:</b> {escape(rec['smiles'])}", body))
        story.append(Spacer(1, 6))
        png = draw_mol(rec["mol"], size=(560, 360))
        story.append(RLImage(io.BytesIO(png), width=3.9 * inch, height=2.5 * inch))
        story.append(Spacer(1, 6))

        keys = [
            "Molecular Formula", "Molecular Weight", "Exact Mass", "LogP", "TPSA",
            "HBD", "HBA", "Rotatable Bonds", "Ring Count", "Aromatic Rings",
            "Heavy Atoms", "Sigma (σ) Bonds", "Pi (π) Bonds", "Fraction Csp3",
            "Molar Refractivity", "QED (drug-likeness score)",
        ]
        rows = [["Property", "Value"]] + [[k, str(p[k])] for k in keys]
        t = Table(rows, colWidths=[2.8 * inch, 2.4 * inch], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor(SRM_BLUE)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, HexColor("#f3f6fb")]),
        ]))
        story += [t, Spacer(1, 8)]

        v = lipinski_violations(p)
        story.append(Paragraph(
            f"<b>Lipinski Rule of Five:</b> {v} violation(s) — "
            + ("passes (drug-like)." if v <= 1 else "fails (poor oral drug-likeness)."), body))
        fgs = functional_groups(rec["mol"])
        story.append(Paragraph(
            "<b>Functional groups:</b> "
            + (", ".join(f"{k} ×{n}" for k, n in fgs.items()) or "none detected"), body))
        story.append(Spacer(1, 4))
        for note in interpret(p):
            story.append(Paragraph("• " + note.replace("**", ""), small))

    doc.build(story)
    return buf.getvalue()


def build_certificate(student, reg, score, total):
    buf = io.BytesIO()
    w, h = landscape(A4)
    c = rl_canvas.Canvas(buf, pagesize=(w, h))
    c.setStrokeColor(HexColor(SRM_BLUE)); c.setLineWidth(6)
    c.rect(25, 25, w - 50, h - 50)
    c.setStrokeColor(HexColor(SRM_GOLD)); c.setLineWidth(2)
    c.rect(38, 38, w - 76, h - 76)
    if os.path.exists(LOGO):
        c.drawImage(LOGO, w / 2 - 1.6 * inch, h - 2.6 * inch, 3.2 * inch, 1.6 * inch, mask="auto")
    c.setFillColor(HexColor(SRM_BLUE))
    c.setFont("Helvetica-Bold", 30)
    c.drawCentredString(w / 2, h - 3.2 * inch, "Certificate of Completion")
    c.setFillColor(colors.black); c.setFont("Helvetica", 15)
    c.drawCentredString(w / 2, h - 3.9 * inch, "This is to certify that")
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(w / 2, h - 4.5 * inch, student or "Student")
    c.setFont("Helvetica", 14)
    c.drawCentredString(w / 2, h - 4.9 * inch, f"Registration No.: {reg or '-'}")
    c.drawCentredString(w / 2, h - 5.4 * inch,
                        "has completed the Cheminformatics Virtual Laboratory")
    c.drawCentredString(w / 2, h - 5.75 * inch,
                        f"with an assessment score of {score}/{total} ({100 * score / total:.0f}%).")
    c.setFont("Helvetica-Oblique", 11)
    c.drawCentredString(w / 2, 62, f"Issued on {datetime.date.today().strftime('%d %B %Y')} • "
                                   "Division of Chemistry, SRM Institute of Science and Technology, Tiruchirappalli")
    c.save()
    return buf.getvalue()


# ============================================================
# SIDEBAR
# ============================================================

if os.path.exists(LOGO):
    st.sidebar.image(LOGO, width="stretch")
st.sidebar.title("🧪 Virtual Lab")
page = st.sidebar.radio("Navigation", PAGES, key="nav")
st.session_state.progress["visited"].add(page)

st.sidebar.divider()
st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")

st.sidebar.divider()
lab_pages = PAGES[1:]
done = len([p for p in st.session_state.progress["visited"] if p in lab_pages])
st.sidebar.markdown("### 🏅 My Progress")
st.sidebar.progress(done / len(lab_pages), text=f"{done}/{len(lab_pages)} modules explored")
st.sidebar.caption(f"Molecules analysed: **{len(st.session_state.progress['molecules'])}**")
best = st.session_state.progress["best"]
st.sidebar.caption("Best quiz score: **" + (f"{best}%" if best is not None else "—") + "**")

# ============================================================
# HOME
# ============================================================

if page == PAGES[0]:
    if os.path.exists(LOGO):
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            st.image(LOGO, width="stretch")
    st.markdown(
        """<div class="hero"><h1>🧪 Cheminformatics Virtual Laboratory</h1>
        <p>SRM Institute of Science and Technology, Tiruchirappalli · Faculty of Engineering and Technology ·
        School of Sciences · Division of Chemistry</p></div>""",
        unsafe_allow_html=True,
    )
    st.write(
        "Represent, visualise and analyse molecules with computational tools — "
        "the same methods used in modern drug discovery and materials research."
    )

    cards = [
        ("🔤 SMILES Playground", "Decode SMILES, inspect atoms & bonds, solve naming challenges."),
        ("🧬 2D & 3D Visualization", "Interactive 3D models, atom labels, bond lengths and angles."),
        ("📊 Properties & Drug-likeness", "20+ descriptors, Lipinski/Veber/Egan rules, QED, radar plot."),
        ("📈 Compare Molecules", "Upload a CSV, compare descriptors and see property charts."),
        ("🔍 Similarity & Substructure", "Fingerprint Tanimoto heat-maps and SMARTS substructure search."),
        ("📝 Assessment", "Randomised quiz with explanations and a completion certificate."),
    ]
    for row in (cards[:3], cards[3:]):
        cols = st.columns(3)
        for col, (t, d) in zip(cols, row):
            with col.container(border=True):
                st.markdown(f"**{t}**")
                st.caption(d)

    st.markdown("### 🚀 Quick start")
    st.markdown(
        "1. Enter your name in the sidebar.\n"
        "2. Start with **Theory**, then try the **SMILES Playground**.\n"
        "3. Analyse a molecule in **Visualization** and **Properties**.\n"
        "4. Take the **Assessment** and download your **Final Report**."
    )
    st.markdown("### Example molecules")
    st.dataframe(
        pd.DataFrame({"Molecule": list(EXAMPLES), "SMILES": list(EXAMPLES.values())}),
        width="stretch", hide_index=True,
    )
    st.divider()
    st.markdown("### 👨‍🏫 Developed by\n**Heman Kowsik M** - Cybersecurity A")


# ============================================================
# THEORY
# ============================================================

elif page == PAGES[1]:
    st.title("📖 Theory")
    t1, t2, t3, t4, t5 = st.tabs(
        ["Cheminformatics", "SMILES rules", "Descriptors", "Drug-likeness", "Fingerprints"]
    )
    with t1:
        st.markdown(
            """
**Cheminformatics** combines chemistry ⚗️, computer science 💻 and data analysis 📊 to store,
search, visualise and predict the properties of molecules.

**Typical applications:** virtual screening, ADMET prediction, QSAR/QSPR modelling,
chemical databases (PubChem, ChEMBL), reaction informatics and materials design.

**Workflow in this lab:** *Text (SMILES) → molecular graph → 2D/3D structure → descriptors → decision.*
"""
        )
    with t2:
        st.markdown("### SMILES — Simplified Molecular Input Line Entry System")
        st.table(pd.DataFrame({
            "Feature": ["Atoms", "Aromatic atoms", "Single / double / triple bond",
                        "Branch", "Ring closure", "Charge / explicit atom",
                        "Stereo (chirality)", "Double-bond geometry"],
            "Notation": ["C, N, O, Cl, Br", "c, n, o (lower case)", "– (implicit)  =  #",
                         "( ... )", "same digit twice: C1CCCCC1", "[NH4+]  [O-]  [Fe+2]",
                         "@  and  @@", "F/C=C/F  vs  F/C=C\\F"],
            "Example": ["CCO", "c1ccccc1", "C=C, C#N", "CC(=O)O", "C1CCCCC1",
                        "[NH4+]", "N[C@@H](C)C(=O)O", "F/C=C/F"],
        }))
        st.info("A **canonical SMILES** is a unique string for a molecule, so the same structure "
                "written differently (OCC, CCO, C(O)C) always gives the same canonical form.")
    with t3:
        st.table(pd.DataFrame({
            "Descriptor": ["Molecular weight", "LogP", "TPSA", "HBD / HBA", "Rotatable bonds",
                           "Fsp3", "QED", "σ and π bonds"],
            "Meaning": [
                "Mass of one mole (g/mol)",
                "Crippen partition coefficient: octanol/water lipophilicity",
                "Topological polar surface area (Å²) of N, O and attached H",
                "Hydrogen-bond donors / acceptors",
                "Single non-ring bonds that allow rotation: flexibility",
                "Fraction of sp³ carbons: 3-D character",
                "Quantitative Estimate of Drug-likeness, 0 (poor) to 1 (ideal)",
                "Each connection = 1 σ bond; double = σ+π; triple = σ+2π",
            ],
        }))
    with t4:
        st.markdown(
            """
| Rule | Criteria |
|---|---|
| **Lipinski (Ro5)** | MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10 (≤ 1 violation acceptable) |
| **Veber** | Rotatable bonds ≤ 10 and TPSA ≤ 140 Å² |
| **Egan** | TPSA ≤ 131.6 Å² and LogP ≤ 5.88 |

These are *guidelines for oral drugs*, not laws — many successful drugs break them.
"""
        )
    with t5:
        st.markdown(
            """
A **molecular fingerprint** converts a structure into a bit-vector. Molecules are compared by the
**Tanimoto coefficient**  T = c / (a + b − c), where *a, b* are the bits set in each molecule and *c* the
bits shared. T ranges from **0** (nothing in common) to **1** (identical fingerprints).

* **Morgan (ECFP):** circular atom environments of a chosen radius.
* **RDKit FP:** linear sub-paths in the molecular graph.
* **MACCS keys:** 166 predefined structural questions.

**SMARTS** is an extension of SMILES for writing *patterns* (e.g. `[CX3](=O)[OX2H1]` = carboxylic acid)
used for substructure searching.
"""
        )

# ============================================================
# SMILES PLAYGROUND
# ============================================================

elif page == PAGES[2]:
    st.title("🔤 SMILES Playground")
    mol, smiles = molecule_input()
    if mol is not None:
        tabs = st.tabs(["Identifiers", "Atoms", "Bonds", "🎯 Challenge"])
        p = props_for(smiles)
        with tabs[0]:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(draw_mol(mol, indices=True), caption="2D structure with atom indices")
            with c2:
                st.markdown(f"**Input SMILES:** `{smiles}`")
                st.markdown(f"**Canonical SMILES:** `{p['Canonical SMILES']}`")
                st.markdown(f"**Isomeric SMILES:** `{Chem.MolToSmiles(mol, isomericSmiles=True)}`")
                st.markdown(f"**Formula:** {p['Molecular Formula']}")
                try:
                    st.markdown(f"**InChI:** `{Chem.MolToInchi(mol)}`")
                except Exception:
                    pass
                st.markdown(f"**InChIKey:** `{p['InChIKey']}`")
                st.markdown(f"**Stereocentres:** {p['Stereocentres']}")
                st.download_button("⬇️ Download MOL file", Chem.MolToMolBlock(mol),
                                   file_name="molecule.mol", mime="chemical/x-mdl-molfile")
        with tabs[1]:
            rows = [{
                "Index": a.GetIdx(), "Element": a.GetSymbol(), "Connections": a.GetDegree(),
                "H count": a.GetTotalNumHs(), "Hybridisation": str(a.GetHybridization()),
                "Aromatic": a.GetIsAromatic(), "In ring": a.IsInRing(),
                "Formal charge": a.GetFormalCharge(),
            } for a in mol.GetAtoms()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with tabs[2]:
            rows = [{
                "Bond": b.GetIdx(),
                "Atoms": f"{b.GetBeginAtom().GetSymbol()}{b.GetBeginAtomIdx()} – "
                         f"{b.GetEndAtom().GetSymbol()}{b.GetEndAtomIdx()}",
                "Type": str(b.GetBondType()), "Aromatic": b.GetIsAromatic(),
                "In ring": b.IsInRing(), "Conjugated": b.GetIsConjugated(),
            } for b in mol.GetBonds()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("σ bonds", p["Sigma (σ) Bonds"])
            m2.metric("π bonds", p["Pi (π) Bonds"])
            m3.metric("Total (σ + π)", p["Sigma (σ) Bonds"] + p["Pi (π) Bonds"])
        with tabs[3]:
            name = st.selectbox("Write the SMILES for:", list(CHALLENGES), key="challenge_name")
            guess = st.text_input("Your SMILES", key="challenge_guess")
            if guess.strip():
                gm = Chem.MolFromSmiles(guess.strip())
                if gm is None:
                    st.error("That is not a valid SMILES.")
                elif Chem.MolToSmiles(gm) == Chem.MolToSmiles(Chem.MolFromSmiles(CHALLENGES[name])):
                    st.success(f"🎉 Correct! {name} = {guess.strip()}")
                    st.image(draw_mol(gm, size=(400, 280)))
                else:
                    st.warning(f"Valid SMILES, but it is **{rdMolDescriptors.CalcMolFormula(gm)}**, "
                               f"not {name}. Try again.")
                    st.image(draw_mol(gm, size=(400, 280)))

# ============================================================
# 2D & 3D VISUALIZATION
# ============================================================

elif page == PAGES[3]:
    st.title("🧬 2D & 3D Visualization")
    mol, smiles = molecule_input()
    if mol is not None:
        tab2d, tab3d, tabgeo = st.tabs(["2D structure", "Interactive 3D", "Geometry explorer"])
        with tab2d:
            idx = st.checkbox("Show atom indices")
            png = draw_mol(mol, size=(800, 520), indices=idx)
            st.image(png)
            st.download_button("⬇️ Download 2D image (PNG)", png, file_name="structure_2d.png",
                               mime="image/png")
        built = build_3d(smiles)
        with tab3d:
            if built is None:
                st.error("Could not generate a 3D structure for this molecule.")
            else:
                molblock, energy, ff = built
                c1, c2, c3, c4 = st.columns(4)
                style = c1.selectbox("Model", ["Ball and Stick", "Stick", "Space Filling", "Wireframe"])
                surface = c2.checkbox("vdW surface")
                labels = c3.checkbox("Atom labels")
                spin = c4.checkbox("Auto-rotate")
                show_3d(molblock, style, surface, labels, spin)
                if energy is not None:
                    st.caption(f"Geometry optimised with **{ff}** · steric energy ≈ **{energy:.2f} kcal/mol** "
                               "(compare conformers/isomers only, not absolute values).")
                st.download_button("⬇️ Download 3D structure (MOL)", molblock,
                                   file_name="structure_3d.mol", mime="chemical/x-mdl-molfile")
        with tabgeo:
            if built is None:
                st.info("3D coordinates are needed for the geometry explorer.")
            else:
                m3 = Chem.MolFromMolBlock(built[0], removeHs=False)
                conf = m3.GetConformer()
                opts = {i: f"{a.GetSymbol()}{i}" for i, a in enumerate(m3.GetAtoms())}
                st.write("Pick atoms to measure (indices shown with *Atom labels* in the 3D tab).")
                a, b, c = st.columns(3)
                i = a.selectbox("Atom A", list(opts), format_func=opts.get, index=0)
                j = b.selectbox("Atom B", list(opts), format_func=opts.get, index=min(1, len(opts) - 1))
                k = c.selectbox("Atom C (optional, for angle)", [None] + list(opts),
                                format_func=lambda x: "—" if x is None else opts[x])
                if i != j:
                    st.metric(f"Distance {opts[i]}–{opts[j]}",
                              f"{rdMolTransforms.GetBondLength(conf, i, j):.3f} Å")
                    if k is not None and k not in (i, j):
                        st.metric(f"Angle {opts[i]}–{opts[j]}–{opts[k]}",
                                  f"{rdMolTransforms.GetAngleDeg(conf, i, j, k):.1f}°")
                else:
                    st.warning("Choose two different atoms.")

# ============================================================
# PROPERTIES & DRUG-LIKENESS
# ============================================================

elif page == PAGES[4]:
    st.title("📊 Properties & Drug-likeness")
    mol, smiles = molecule_input()
    if mol is not None:
        p = props_for(smiles)
        m = st.columns(5)
        m[0].metric("Mol. weight", p["Molecular Weight"])
        m[1].metric("LogP", p["LogP"])
        m[2].metric("TPSA (Å²)", p["TPSA"])
        m[3].metric("HBD / HBA", f"{p['HBD']} / {p['HBA']}")
        m[4].metric("QED", p["QED (drug-likeness score)"])

        t1, t2, t3, t4 = st.tabs(["Descriptors", "Drug-likeness", "Functional groups", "Interpretation"])
        with t1:
            c1, c2 = st.columns([1, 1.4])
            c1.image(draw_mol(mol, size=(480, 360)))
            df = pd.DataFrame({"Property": list(p), "Value": [str(v) for v in p.values()]})
            c2.dataframe(df, width="stretch", hide_index=True, height=380)
            st.download_button("⬇️ Download descriptors (CSV)", df.to_csv(index=False).encode(),
                               file_name="descriptors.csv", mime="text/csv")
        with t2:
            rules = drug_likeness(p)
            c1, c2 = st.columns([1.3, 1])
            with c1:
                for rule, items in rules.items():
                    ok = all(x[2] for x in items)
                    st.markdown(
                        f"**{rule}** — "
                        + ('<span class="pill-ok">PASS</span>' if ok else '<span class="pill-bad">FAIL</span>'),
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame([{"Criterion": c, "Value": v, "Result": "✅" if o else "❌"}
                                      for c, v, o in items]),
                        width="stretch", hide_index=True,
                    )
                v = lipinski_violations(p)
                (st.success if v <= 1 else st.warning)(
                    f"Lipinski violations: **{v}** — "
                    + ("good oral drug-likeness." if v <= 1 else "likely poor oral absorption."))
            with c2:
                labels = ["MW/500", "LogP/5", "HBD/5", "HBA/10", "RotB/10", "TPSA/140"]
                vals = [p["Molecular Weight"] / 500, p["LogP"] / 5, p["HBD"] / 5,
                        p["HBA"] / 10, p["Rotatable Bonds"] / 10, p["TPSA"] / 140]
                vals = [max(0, x) for x in vals]
                ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                fig = plt.figure(figsize=(4.4, 4.4))
                ax = fig.add_subplot(111, polar=True)
                ax.plot(ang + ang[:1], [1] * (len(ang) + 1), "--", color="#c8102e", lw=1, label="Limit")
                ax.plot(ang + ang[:1], vals + vals[:1], color=SRM_BLUE, lw=2)
                ax.fill(ang + ang[:1], vals + vals[:1], color=SRM_BLUE, alpha=0.25)
                ax.set_xticks(ang)
                ax.set_xticklabels(labels, fontsize=9)
                ax.set_ylim(0, max(1.5, max(vals) * 1.1))
                ax.set_title("Property radar (red dashes = limit)", fontsize=10)
                st.pyplot(fig)
                plt.close(fig)
        with t3:
            fgs = functional_groups(mol)
            if fgs:
                st.dataframe(pd.DataFrame({"Functional group": list(fgs), "Count": list(fgs.values())}),
                             width="stretch", hide_index=True)
                pick = st.selectbox("Highlight a group in the structure", list(fgs))
                hl = {a for match in mol.GetSubstructMatches(FG_PATTERNS[pick]) for a in match}
                st.image(draw_mol(mol, highlight=hl, size=(600, 400)))
            else:
                st.info("No common functional groups detected (e.g. a simple hydrocarbon).")
        with t4:
            for note in interpret(p):
                st.markdown("• " + note)
        pdf = build_report_pdf(student_name, registration_number,
                               [{"name": "Molecule", "smiles": smiles, "mol": mol}])
        st.download_button("📄 Download PDF report for this molecule", pdf,
                           file_name="molecule_report.pdf", mime="application/pdf")

# ============================================================
# COMPARE MOLECULES
# ============================================================

elif page == PAGES[5]:
    st.title("📈 Compare Molecules")
    src = st.radio("Input", ["Type / paste", "Upload CSV"], horizontal=True)
    text = DEFAULT_LIBRARY
    if src == "Type / paste":
        text = st.text_area("One molecule per line:  Name, SMILES", value=DEFAULT_LIBRARY, height=200)
    else:
        up = st.file_uploader("CSV with columns `name` and `smiles`", type="csv")
        if up is not None:
            try:
                raw = pd.read_csv(up)
                raw.columns = [c.strip().lower() for c in raw.columns]
                if "smiles" not in raw.columns:
                    st.error("The CSV needs a 'smiles' column.")
                    text = ""
                else:
                    names = raw["name"] if "name" in raw.columns else [f"Molecule {i + 1}" for i in range(len(raw))]
                    text = "\n".join(f"{n},{s}" for n, s in zip(names, raw["smiles"]))
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                text = ""
        else:
            st.info("Upload a file, or switch to *Type / paste*.")
            text = ""

    recs, bad = parse_library(text)
    if bad:
        st.warning("Skipped invalid SMILES: " + ", ".join(bad))
    if recs:
        rows = []
        for r in recs:
            pr = props_for(r["smiles"])
            st.session_state.progress["molecules"].add(pr["Canonical SMILES"])
            rows.append({
                "Molecule": r["name"], "Formula": pr["Molecular Formula"],
                "MW": pr["Molecular Weight"], "LogP": pr["LogP"], "TPSA": pr["TPSA"],
                "HBD": pr["HBD"], "HBA": pr["HBA"], "RotB": pr["Rotatable Bonds"],
                "Rings": pr["Ring Count"], "QED": pr["QED (drug-likeness score)"],
                "Ro5 violations": lipinski_violations(pr),
            })
        df = pd.DataFrame(rows)
        tt, tc, tg = st.tabs(["Table", "Charts", "Structure gallery"])
        with tt:
            st.dataframe(df, width="stretch", hide_index=True)
            st.download_button("⬇️ Download comparison (CSV)", df.to_csv(index=False).encode(),
                               file_name="comparison.csv", mime="text/csv")
        with tc:
            num_cols = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "Rings", "QED"]
            c1, c2 = st.columns(2)
            with c1:
                prop = st.selectbox("Bar chart property", num_cols)
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.bar(df["Molecule"], df[prop], color=SRM_BLUE)
                ax.set_ylabel(prop)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with c2:
                st.markdown("**Permeability map** — TPSA vs LogP")
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.axvspan(0, 131.6, ymin=0, ymax=1, color="#dff5e3", alpha=0.6)
                ax.axhline(5.88, color="#c8102e", ls="--", lw=1)
                ax.scatter(df["TPSA"], df["LogP"], s=70, color=SRM_BLUE, zorder=3)
                for _, r in df.iterrows():
                    ax.annotate(r["Molecule"], (r["TPSA"], r["LogP"]), fontsize=8,
                                xytext=(4, 4), textcoords="offset points")
                ax.set_xlabel("TPSA (Å²)")
                ax.set_ylabel("LogP")
                ax.set_xlim(left=0)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Green zone: TPSA ≤ 131.6 Å²; dashed line: LogP = 5.88 (Egan limits).")
        with tg:
            grid = Draw.MolsToGridImage(
                [r["mol"] for r in recs], molsPerRow=4, subImgSize=(260, 210),
                legends=[r["name"] for r in recs],
            )
            st.image(grid)
    else:
        st.info("Enter at least one valid molecule.")

# ============================================================
# SIMILARITY & SUBSTRUCTURE
# ============================================================

elif page == PAGES[6]:
    st.title("🔍 Similarity & Substructure Search")
    text = st.text_area("Molecule library (Name, SMILES per line)", value=DEFAULT_LIBRARY, height=180)
    recs, bad = parse_library(text)
    if bad:
        st.warning("Skipped invalid SMILES: " + ", ".join(bad))

    if len(recs) < 2:
        st.info("Add at least two valid molecules.")
    else:
        ts, tq = st.tabs(["Tanimoto similarity", "Substructure (SMARTS) search"])
        with ts:
            fp_type = st.selectbox("Fingerprint", ["Morgan (radius 2)", "RDKit path FP", "MACCS keys"])
            if fp_type.startswith("Morgan"):
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fps = [gen.GetFingerprint(r["mol"]) for r in recs]
            elif fp_type.startswith("RDKit"):
                fps = [Chem.RDKFingerprint(r["mol"]) for r in recs]
            else:
                fps = [MACCSkeys.GenMACCSKeys(r["mol"]) for r in recs]
            n = len(recs)
            sim = np.array([[DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in range(n)]
                            for i in range(n)])
            names = [r["name"] for r in recs]
            fig, ax = plt.subplots(figsize=(min(9, 1 + 0.8 * n), min(7, 0.5 + 0.7 * n)))
            im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
            if n <= 12:
                for i in range(n):
                    for j in range(n):
                        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if sim[i, j] > 0.6 else "black")
            fig.colorbar(im, ax=ax, label="Tanimoto")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            q = st.selectbox("Rank all molecules against:", names)
            qi = names.index(q)
            rank = pd.DataFrame({"Molecule": names, "Tanimoto vs query": sim[qi].round(3)})
            rank = rank[rank["Molecule"] != q].sort_values("Tanimoto vs query", ascending=False)
            st.dataframe(rank, width="stretch", hide_index=True)
            st.caption("T ≥ 0.7 usually means closely related structures (Morgan FP); values depend on the fingerprint.")

        with tq:
            presets = {"Custom": "", **FUNCTIONAL_GROUPS}
            choice = st.selectbox("Preset pattern", list(presets))
            smarts = st.text_input("SMARTS pattern", value=presets[choice] or "c1ccccc1",
                                   key=f"smarts_{choice}")
            patt = Chem.MolFromSmarts(smarts) if smarts.strip() else None
            if patt is None:
                st.error("Invalid SMARTS pattern.")
            else:
                hits = [(r, r["mol"].GetSubstructMatch(patt)) for r in recs if r["mol"].HasSubstructMatch(patt)]
                st.write(f"**{len(hits)} of {len(recs)}** molecules contain this pattern.")
                if hits:
                    grid = Draw.MolsToGridImage(
                        [r["mol"] for r, _ in hits], molsPerRow=3, subImgSize=(280, 220),
                        legends=[r["name"] for r, _ in hits],
                        highlightAtomLists=[list(m) for _, m in hits],
                    )
                    st.image(grid)

# ============================================================
# ASSESSMENT
# ============================================================

elif page == PAGES[7]:
    st.title("📝 Cheminformatics Assessment")

    BANK = [
        ("What is the full form of SMILES?",
         ["Simplified Molecular Input Line Entry System", "Standard Molecular Information Language Encoding System",
          "Simple Molecular Identification and Labeling System"], 0,
         "SMILES encodes a molecular graph as a line of text."),
        ("Which SMILES represents benzene?", ["c1ccccc1", "CCO", "O=C=O"], 0,
         "Lower-case c marks aromatic carbons; the digit 1 closes the ring."),
        ("In SMILES, parentheses indicate:", ["Branches", "Charges", "Stereochemistry"], 0,
         "CC(=O)O: the =O is a branch off the second carbon."),
        ("Which symbol denotes a triple bond in SMILES?", ["#", "=", "%"], 0, "'=' is double, '#' is triple."),
        ("Which property is the usual measure of lipophilicity?", ["LogP", "TPSA", "HBA"], 0,
         "LogP is the octanol/water partition coefficient."),
        ("TPSA stands for:", ["Topological Polar Surface Area", "Total Pi Surface Area",
                              "Thermal Property Surface Analysis"], 0,
         "TPSA sums the surface contributions of polar atoms (N, O and their H)."),
        ("Lipinski's Rule of Five allows a maximum molecular weight of:", ["500", "300", "1000"], 0,
         "Ro5: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10."),
        ("Under Lipinski's rule the maximum number of H-bond donors is:", ["5", "10", "2"], 0,
         "HBD ≤ 5 and HBA ≤ 10."),
        ("How many π bonds are in a C≡C triple bond?", ["2", "1", "3"], 0, "Triple = 1 σ + 2 π."),
        ("How many σ bonds are present in methane (CH₄)?", ["4", "5", "2"], 0, "Four C–H single bonds, each a σ bond."),
        ("Ethene (C₂H₄) contains:", ["5 σ and 1 π bond", "4 σ and 2 π bonds", "6 σ and 0 π bonds"], 0,
         "4 C–H σ + 1 C–C σ + 1 C=C π."),
        ("Two different SMILES for the same molecule give the same:", ["Canonical SMILES", "Atom order",
                                                                         "Ring-closure digits"], 0,
         "Canonicalisation produces one unique string per structure."),
        ("The Tanimoto coefficient ranges from:", ["0 to 1", "−1 to 1", "0 to 100 000"], 0,
         "0 = no shared bits, 1 = identical fingerprints."),
        ("Morgan (ECFP) fingerprints encode:", ["Circular atom environments", "3D shape only", "Isotope ratios"], 0,
         "Each bit represents an atom neighbourhood within the chosen radius."),
        ("SMARTS is mainly used for:", ["Substructure pattern searching", "Drawing 3D models", "Calculating MW"], 0,
         "SMARTS patterns describe substructures to search for."),
        ("The Veber rule uses rotatable bonds ≤ 10 and:", ["TPSA ≤ 140 Å²", "LogP ≤ 5", "MW ≤ 500"], 0,
         "Veber relates oral bioavailability to flexibility and polarity."),
        ("Which 3D model best shows approximate molecular volume?", ["Space filling", "Wireframe", "Text SMILES"], 0,
         "Space-filling spheres use van der Waals radii."),
        ("Fsp3 is the fraction of:", ["sp³-hybridised carbons", "sp²-hybridised carbons", "sp-hybridised atoms"], 0,
         "Higher Fsp3 means more three-dimensional, less flat molecules."),
        ("In SMILES, [NH4+] shows:", ["An explicit atom with a charge", "A ring closure", "A double bond"], 0,
         "Square brackets specify atom, hydrogen count and charge explicitly."),
        ("QED values close to 1 indicate:", ["High drug-likeness", "Low molecular weight only", "Toxicity"], 0,
         "QED combines several desirability functions into one score."),
    ]
    N_Q = 10

    if "quiz_attempt" not in st.session_state:
        st.session_state.quiz_attempt = 0

    def new_quiz():
        st.session_state.quiz_attempt += 1
        st.session_state.pop("quiz_result", None)
        rng = random.Random()
        items = []
        for q, opts, ans, why in rng.sample(BANK, N_Q):
            correct = opts[ans]
            shuffled = opts[:]
            rng.shuffle(shuffled)
            items.append({"q": q, "opts": shuffled, "a": correct, "why": why})
        st.session_state.quiz_items = items

    if "quiz_items" not in st.session_state:
        new_quiz()

    st.caption(f"{N_Q} questions drawn at random from a bank of {len(BANK)}. Pass mark: 60 %.")
    att = st.session_state.quiz_attempt
    with st.form(f"quiz_form_{att}"):
        answers = [
            st.radio(f"**Q{i + 1}.** {it['q']}", it["opts"], index=None, key=f"q_{att}_{i}")
            for i, it in enumerate(st.session_state.quiz_items)
        ]
        submitted = st.form_submit_button("Submit assessment")

    if submitted:
        if any(a is None for a in answers):
            st.warning("Please answer every question before submitting.")
        else:
            score = sum(a == it["a"] for a, it in zip(answers, st.session_state.quiz_items))
            pct = round(100 * score / N_Q)
            st.session_state.quiz_result = {"score": score, "answers": answers, "pct": pct}
            prev = st.session_state.progress["best"]
            st.session_state.progress["best"] = pct if prev is None else max(prev, pct)

    res = st.session_state.get("quiz_result")
    if res:
        st.metric("Your score", f"{res['score']}/{N_Q}", f"{res['pct']}%")
        if res["pct"] >= 80:
            st.success("🎉 Excellent performance!")
        elif res["pct"] >= 60:
            st.info("👍 Good — you passed. Review the explanations below.")
        else:
            st.warning("📚 Please revisit the Theory section and try a new set.")
        with st.expander("Review answers and explanations", expanded=True):
            for i, (a, it) in enumerate(zip(res["answers"], st.session_state.quiz_items), 1):
                ok = a == it["a"]
                st.markdown(f"{'✅' if ok else '❌'} **Q{i}.** {it['q']}")
                st.markdown(f"Your answer: {a}" + ("" if ok else f" · Correct: **{it['a']}**"))
                st.caption(it["why"])
        if res["pct"] >= 60:
            st.download_button(
                "🎓 Download certificate (PDF)",
                build_certificate(student_name, registration_number, res["score"], N_Q),
                file_name="Cheminformatics_Lab_Certificate.pdf", mime="application/pdf",
            )
            if not student_name:
                st.caption("Tip: enter your name in the sidebar first so it appears on the certificate.")
    st.button("🔄 New question set", on_click=new_quiz)

# ============================================================
# FINAL REPORT
# ============================================================

elif page == PAGES[8]:
    st.title("📄 Final Project Report")
    st.write("Enter your molecules (one per line, `Name, SMILES` or just SMILES). "
             "All valid molecules go into one PDF with structures, descriptors, drug-likeness and interpretation.")
    c1, c2 = st.columns(2)
    c1.text_input("Student Name", value=student_name, disabled=True,
                  help="Edit in the sidebar.")
    c2.text_input("Registration Number", value=registration_number, disabled=True,
                  help="Edit in the sidebar.")
    default = f"Active molecule,{st.session_state.active_smiles}"
    text = st.text_area("Molecules", value=default, height=170, key="report_text")
    recs, bad = parse_library(text)
    if bad:
        st.warning("These lines could not be read and will be skipped: " + ", ".join(bad))
    if recs:
        st.success(f"{len(recs)} valid molecule(s) ready.")
        prev = pd.DataFrame([{
            "No.": i, "Name": r["name"], "SMILES": r["smiles"],
            "Formula": props_for(r["smiles"])["Molecular Formula"],
            "MW": props_for(r["smiles"])["Molecular Weight"],
        } for i, r in enumerate(recs, 1)])
        st.dataframe(prev, width="stretch", hide_index=True)
        if not student_name:
            st.info("Add your name and registration number in the sidebar to personalise the report.")
        if st.button("📄 Generate PDF report"):
            with st.spinner("Building report..."):
                st.session_state.report_pdf = build_report_pdf(student_name, registration_number, recs)
        if st.session_state.get("report_pdf"):
            st.download_button("⬇️ Download report (PDF)", st.session_state.report_pdf,
                               file_name="Cheminformatics_Report.pdf", mime="application/pdf")
    else:
        st.info("Enter at least one valid SMILES string.")

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("🧪 Cheminformatics Virtual Laboratory v2.0 · Division of Chemistry, SRM IST Tiruchirappalli · "
           "Built with Python, Streamlit and RDKit")
