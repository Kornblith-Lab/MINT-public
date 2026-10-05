"""
build_images.py — Render human-review trajectories as an easy-to-read PDF.

Turns each trajectory produced by ``fig1_excel.py`` into one page of a multi-page
PDF so a human reviewer can look at a case and predict what happens next. Pages are
in the SAME order as the rows of the Excel workbook, so page N corresponds to row N
of the ``model`` sheet used by ``llm.py``.

The page is blinded: it shows only the trajectory (the ``human`` sheet), never the
``label`` or the ``mint`` probability. (We may add optional MINT-probability hints
here later — see the note in ``main``.)

Layout (top to bottom, all panels share one time axis of minutes since arrival):
  1. Triage header  — mode of arrival, age, sex, ESI/acuity, chief complaint, weight
  2. SpO2           — line
  3. Heart rate     — line   (Vital_Pulse)
  4. Resp rate      — line   (Vital_Resp)
  5. Blood pressure — points (Systolic / Diastolic)
  6. Temp + GCS     — points (two colors, twin y-axis)
  7. O2 support     — device timeline (step) + O2 flow rate (points, twin axis)
  8. Medications    — vertical lines at administration times (drug name; no dose)
  9. Procedures/Labs— vertical lines at order times (labs shown; all labs abnormal)
 10. Leftover       — any token not covered above, listed with its timestamp

A vertical "Now" line marks the final token; the shaded band to its right is the
``--lookahead`` prediction window the reviewer is asked about.

EVERY token in a trajectory is routed to exactly one bucket above; a per-page
assertion guarantees nothing is silently dropped (unroutable tokens fall into
"Leftover").

Each PDF starts with a title page. The base cover appears on all outputs; the
MINT cover is inserted after it only on PDFs that also show MINT predicted
probabilities.

Each page carries an interactive, fillable "Risk ratio for PPV:" text field next
to the "Case N" title (requires reportlab + pypdf). Reviewers type a value directly
in any PDF viewer that supports forms (Acrobat, Preview, Edge, browsers); the value
is a per-page named field (``risk_ratio_page_{N}``) that persists when they save the
file. Collect the responses afterwards with pypdf, e.g.::

    from pypdf import PdfReader
    fields = PdfReader("ppv_review.pdf").get_fields()
    answers = {k: v.get("/V") for k, v in fields.items()}  # page index -> value

If reportlab/pypdf are not installed, the PDF is still written but without fields.

--------------------------------------------------------------------------------
Usage (standalone)
--------------------------------------------------------------------------------
    python -m mint.five.fig_one.comparator.build_images \
        --excel artifacts/fig1/human_review/ppv.xlsx

    # infer the prediction question from a mode different from the filename, and
    # match the lookahead used when the workbook was built:
    python -m mint.five.fig_one.comparator.build_images \
        --excel artifacts/fig1/human_review/ppv.xlsx --mode ppv --lookahead 5

    # quick preview of the first few pages:
    python -m mint.five.fig_one.comparator.build_images \
        --excel artifacts/fig1/human_review/ppv.xlsx --max_pages 5

Writes ``<excel stem>_review.pdf`` next to the workbook (override with --out).

--------------------------------------------------------------------------------
Full comparator pipeline
--------------------------------------------------------------------------------
  1. fig1_excel.py   Sample K test cases for a task, run MINT, and write
                     artifacts/fig1/human_review/<mode>.xlsx with sheets
                     "human" (trajectory only) and "model" (trajectory+label+mint).

       python -m mint.five.fig_one.comparator.fig1_excel --mode ppv --lookahead 5

  2. build_images.py Render the blinded trajectories to <mode>_review.pdf. A human
                     reviews page N, decides P(outcome), and records it in the
                     "human" column of the workbook (row N). PAGE ORDER == ROW ORDER.

       python -m mint.five.fig_one.comparator.build_images \
           --excel artifacts/fig1/human_review/ppv.xlsx --mode ppv --lookahead 5

  3. llm.py          Run the LLM comparator on the same rows and print/plot
                     AUROC/AUPRC for mint vs. llm vs. human (if the human column
                     was filled in).

       python -m mint.five.fig_one.comparator.llm \
           --excel artifacts/fig1/human_review/ppv.xlsx
"""

import io
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tap import tapify

# Okabe-Ito colorblind-safe palette (matches design-skill / nature.mplstyle).
C_SPO2 = "#0072B2"      # blue
C_HR = "#D55E00"        # vermillion
C_RESP = "#009E73"      # green
C_SYS = "#D55E00"       # vermillion
C_DIA = "#56B4E9"       # sky blue
C_MAP = "#009E73"       # green
C_TEMP = "#E69F00"      # orange
C_GCS = "#CC79A7"       # purple
C_MED = "#CC79A7"       # purple
C_PROC = "#0072B2"      # blue
C_DEVICE = "#000000"    # black
C_FLOW = "#56B4E9"      # sky blue
C_NOW = "#999999"       # grey

# Units for the small per-point value annotations on each numeric panel.
UNITS = {
    "SpO2": "%", "Pulse": " bpm", "Resp": "/min",
    "Systolic": "", "Diastolic": "", "MAP": " mmHg",
    "Temp": " F", "GCS": "",
}

# O2 device severity (higher = higher acuity), keyed by the device text stored in
# ParsedCase.devices (i.e. with the "Vital_O2 Device_" prefix stripped). Sourced
# from mint.model.respiratory.RespiratoryEscalation.SEVERITY. We load that module
# directly by file path to avoid importing the whole mint.model package (which
# pulls in torch); this script only needs to read an xlsx.
import importlib.util as _ilu

_resp_path = Path(__file__).resolve().parents[3] / "model" / "respiratory.py"
_spec = _ilu.spec_from_file_location("_mint_respiratory", _resp_path)
_resp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_resp)
DEVICE_SEVERITY = {
    k[len("Vital_O2 Device_"):]: v
    for k, v in _resp.RespiratoryEscalation.SEVERITY.items()
}

# Prediction question per task mode (see mint/five/fig_one/fig1.py Definition).
QUESTIONS = {
    "hypoxia": "Will this child become hypoxic (SpO2 <= 92%) in the next {la} to 60 minutes?",
    "tachypnea": "Will this child become tachypneic in the next {la} to 60 minutes?",
    "bradypnea": "Will this child become bradypneic in the next {la} to 60 minutes?",
    "tachycardia": "Will this child become tachycardic in the next {la} to 60 minutes?",
    "periarrest": "Will this child become bradycardic (peri-arrest) in the next {la} to 60 minutes?",
    "hypotension": "Will this child become hypotensive in the next {la} to 60 minutes?",
    "ventilator": "Will this child be placed on a ventilator in the next {la} to 60 minutes?",
    "ppv": "PPV is defined as high-flow nasal cannula, CPAP/BiPAP, BVM, or Ventilator.",
    "resp_rescue": "Will this child receive a respiratory rescue medication in the next {la} to 60 minutes?",
    "vasopressor": "Will this child receive a vasopressor in the next {la} to 60 minutes?",
    "transfusion": "Will this child receive a blood transfusion in the next {la} to 60 minutes?",
    "narcan": "Will this child receive naloxone in the next {la} to 60 minutes?",
    "cardio_rescue": "Will this child receive a cardiac rescue medication in the next {la} to 60 minutes?",
}

BASE_COVER = """
<font size="20"><b>Positive pressure ventilation review guide</b></font><br/><br/>
<font size="13">
You are performing a forecasting task to predict whether a given child will be
initiated on positive pressure ventilation in the next 5 to 60 minutes after a
cutoff timepoint. You will make predictions by providing a number as a
probability in a fillable box at the top right corner of every page. These
probabilities will not be graded directly on their value but instead will be used
to assemble ranks for your cohort. Your ranking performance will then be compared
to artificial intelligence (AI) approaches after you complete your reviews.<br/><br/>
<b>Important notes:</b><br/><br/>
1. The outcome incidence in this cohort of children has been upsampled to between
25% and 30% of children being initiated on PPV. You should expect between 6 to 8
of the children to actually have been initiated on PPV out of the 25.<br/>
2. Every page is a different case from a different child, you will never review
the same visit more than once.<br/>
3. For this task, PPV is defined as high-flow nasal cannula, CPAP, BiPAP, BVM, or
mechanical ventilation. Any one of these in the 5 to 60 minute window (greyed out
area) counts as a positive outcome (&ldquo;Yes&rdquo; for initiation on PPV).<br/>
&nbsp;&nbsp;&nbsp;a. Reminder: HFNC is included as PPV for this study.<br/><br/>
If you are unsure about your predictions, that is okay! We understand that this
will likely be a very difficult prediction task.<br/><br/>
If you have any questions, please reach out to redacted and redacted on
Slack.<br/><br/>
<b>Saving predictions:</b><br/><br/>
Each page contains an editable form in the top right, you should be able to add
predictions here the same way as filling out any other administrative form. Please
be sure to regularly save your work (Ctrl+S or Cmd+S) to ensure that it works
well. We recommend using the default &ldquo;Preview&rdquo; application on Mac or
Adobe Acrobat.
</font>
"""

MINT_COVER = """
<font size="20"><b>Additional instructions for using MINT predictions</b></font><br/><br/>
<font size="13">
This version of the PDF contains additional information which includes the MINT
model&rsquo;s own predicted probability and explanations of risk factors. The MINT
probability indicates how likely the model felt PPV initiation in the next 5-60
minutes was.<br/><br/>
This information is provided to evaluate how collaborating with the MINT model
may improve or worsen your predictive performance as a physician.<br/><br/>
We will leave it up to you on how much importance you place on the MINT
predictions and explanations. The only thing we ask is that you <b>do not just
copy-paste the same value as MINT</b>, since the MINT model will be alone in a
different evaluation.<br/><br/>
The goal is to have you use MINT to refine your own suspicion of risk.<br/><br/>
Therefore, our final evaluation will have three comparisons on every child:<br/><br/>
MINT model only<br/>
Physician only<br/>
Physician + MINT (this task)<br/><br/>
Note: While you are reviewing these children with MINT, a different physician is
reviewing the same children without MINT to create a case-control.
</font>
"""

TOKEN_RE = re.compile(r"^(?P<name>.*) \(t=(?P<t>-?\d+) min\)$")


def parse_trajectory(traj: str) -> list[tuple[str, int]]:
    """'Name (t=X min) -> Name (t=Y min)' -> [(name, X), (name, Y), ...]."""
    events = []
    for part in traj.split(" -> "):
        m = TOKEN_RE.match(part.strip())
        if m:
            events.append((m.group("name"), int(m.group("t"))))
        else:
            # Keep unparseable text so it still surfaces in the leftover list.
            events.append((part.strip(), None))
    return events


def _num(value: str) -> Optional[float]:
    """Pull the leading number out of a vital value like '16kg' or '98'."""
    m = re.search(r"-?\d+\.?\d*", value)
    return float(m.group()) if m else None


def _split(name: str) -> tuple[str, str]:
    """Split 'Vital_MAP (mmHg)_65' -> ('Vital_MAP (mmHg)', '65')."""
    cat, _, val = name.rpartition("_")
    return cat, val


class ParsedCase:
    """Route every token of one trajectory into render buckets (no token dropped)."""

    def __init__(self, traj: str):
        self.events = parse_trajectory(traj)
        self.n_total = len(self.events)

        # Header (static, t=0) fields.
        self.arrival = self.age = self.sex = self.acuity = self.cc = None
        self.weights: list[float] = []

        # Time series: {category: [(t, value)]}.
        self.series: dict[str, list[tuple[int, float]]] = {}
        self.devices: list[tuple[int, str]] = []       # (t, device text)
        self.flows: list[tuple[int, float]] = []        # (t, L/min)
        self.meds: list[tuple[int, str]] = []           # (t, drug)
        self.procs: list[tuple[int, str]] = []          # (t, procedure/lab label)
        self.leftover: list[tuple[Optional[int], str]] = []

        self.n_plotted = 0
        for name, t in self.events:
            self._route(name, t)

        # Coverage contract: every token is either header/plotted or leftover.
        assert self.n_plotted + len(self.leftover) == self.n_total, (
            f"coverage mismatch: {self.n_plotted}+{len(self.leftover)} != {self.n_total}"
        )

    # Numeric-vital prefix -> series key.
    _SERIES = {
        "Vital_SpO2_": "SpO2",
        "Vital_Pulse_": "Pulse",
        "Vital_Resp_": "Resp",
        "Vital_Systolic_": "Systolic",
        "Vital_Diastolic_": "Diastolic",
        "Vital_MAP (mmHg)_": "MAP",
        "Vital_Temp_": "Temp",
        "Vital_Glasgow Coma Scale Score_": "GCS",
    }

    def _route(self, name: str, t: Optional[int]):
        """Send one token to exactly one bucket, keeping n_plotted in lockstep."""
        # Static header tokens (routed regardless of timestamp).
        if name.startswith("Arrival_Method_"):
            self.arrival = name[len("Arrival_Method_"):]
        elif name.startswith("CC_"):
            self.cc = name[len("CC_"):]
        elif name.startswith("Age_"):
            self.age = name[len("Age_"):]
        elif name.startswith("Acuity_"):
            self.acuity = name[len("Acuity_"):]
        elif name.startswith("Sex_"):
            self.sex = name[len("Sex_"):]
        elif name.startswith("Vital_Weight_"):
            v = _num(_split(name)[1])
            if v is not None:
                self.weights.append(v)
            else:
                self.leftover.append((t, name))
                return
        # Everything below needs a timestamp and a parseable value to be plotted.
        elif t is None:
            self.leftover.append((t, name))
            return
        elif any(name.startswith(p) for p in self._SERIES):
            prefix = next(p for p in self._SERIES if name.startswith(p))
            v = _num(_split(name)[1])
            if v is None:
                self.leftover.append((t, name))
                return
            self.series.setdefault(self._SERIES[prefix], []).append((t, v))
        elif name.startswith("Vital_O2 Device_"):
            self.devices.append((t, name[len("Vital_O2 Device_"):]))
        elif name.startswith("Vital_O2 Flow Rate (l/min)_"):
            v = _num(_split(name)[1])
            if v is None:
                self.leftover.append((t, name))
                return
            self.flows.append((t, v))
        elif name.startswith("Med_"):
            self.meds.append((t, name[len("Med_"):]))
        elif name.startswith("Procedure_"):
            self.procs.append((t, name[len("Procedure_"):]))
        elif name.startswith("Lab_"):
            self.procs.append((t, name[len("Lab_"):] + " [lab]"))
        else:
            # Admit / Discharge / ICU Start and anything unmapped.
            self.leftover.append((t, name))
            return
        self.n_plotted += 1

    @property
    def t_now(self) -> int:
        times = [t for _, t in self.events if t is not None]
        return max(times) if times else 0


def _plot_series(ax, data, color, label, connect, unit=""):
    """Plot one time series as a line (connect=True) or points (connect=False).

    Annotates each point with its numeric value (and unit) in small text.
    """
    if not data:
        return
    data = sorted(data)
    xs = [t for t, _ in data]
    ys = [v for _, v in data]
    if connect:
        ax.plot(xs, ys, "-o", color=color, label=label, markersize=6, linewidth=2)
    else:
        ax.plot(xs, ys, "o", color=color, label=label, markersize=8)
    # Label above the dot. clip_on=False lets labels near the top edge overflow
    # the axes box rather than being clipped.
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:g}{unit}", (x, y), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=color, clip_on=False)


def _vline_track(ax, items, color, xmax, fontsize=8):
    """Vertical dashed lines for events (meds / procedures / labs).

    Events sharing a timestamp are collapsed to a single dashed line whose events
    are listed in one stacked (multi-line) label — this is what makes dense lab
    clusters (many results ordered at once) legible. Distinct-time clusters are
    then packed into vertical slots via a greedy assignment: each label goes to the
    lowest slot whose previous label does not horizontally overlap it. Label
    x-extent is estimated from the longest line's character count. The occupied
    slots are finally spread evenly across the full panel height (ymin->ymax) to
    maximise vertical separation between overlapping labels.
    """
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    if not items:
        ax.text(0.5, 0.5, "none", transform=ax.transAxes, ha="center",
                va="center", color="grey", fontsize=11, style="italic")
        return

    # Group events by timestamp -> one line + one stacked label per unique time.
    by_time: dict[int, list[str]] = {}
    for t, name in sorted(items):
        by_time.setdefault(t, []).append(name)

    # Approximate data units per character: axes span ~0.84 of a 16" figure.
    axes_pt = 0.84 * 16 * 72
    dx_per_char = (fontsize * 0.55) * (xmax / axes_pt)
    pad = 1.5 * dx_per_char  # gap between neighbouring labels in the same slot

    # Pass 1: assign each cluster a collision-free slot index (top-packed).
    slots_right_edge: list[float] = []  # rightmost occupied x per slot
    placed: list[tuple[int, str, bool, int]] = []  # (t, label, right, slot)
    for t, names in sorted(by_time.items()):
        label = "\n".join(names)
        # Labels lean left of the line in the right portion of the panel to keep
        # text near the axes; clip_on=False lets long labels overflow the box edge
        # (into the margin) rather than being cut off.
        right = t > 0.6 * xmax
        width = (max(len(n) for n in names) + 2) * dx_per_char
        lo = t - width if right else t
        hi = t if right else t + width
        slot = next((s for s in range(len(slots_right_edge))
                     if lo > slots_right_edge[s] + pad), None)
        if slot is None:
            slot = len(slots_right_edge)
            slots_right_edge.append(hi)
        else:
            slots_right_edge[slot] = hi
        placed.append((t, label, right, slot))

    # Pass 2: spread the occupied slots across the full height (top slot near
    # ymax, bottom slot near ymin) so overlapping labels are maximally separated.
    n_slots = len(slots_right_edge)
    y_top, y_bottom = 0.95, 0.12
    for t, label, right, slot in placed:
        y = y_top if n_slots == 1 else y_top - (y_top - y_bottom) * slot / (n_slots - 1)
        ax.axvline(t, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.annotate(label, (t, y), xytext=(-3 if right else 3, 0),
                    textcoords="offset points", fontsize=fontsize, color=color,
                    ha="right" if right else "left", va="top", clip_on=False,
                    linespacing=1.1)


def _build_cover_pdf(page_size: tuple[float, float], with_mint: bool) -> io.BytesIO:
    """Build one or two formatted title pages as a PDF in memory."""
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "cover_body",
        parent=styles["BodyText"],
        alignment=TA_LEFT,
        fontName="Helvetica",
        fontSize=13,
        leading=16,
    )

    cover_pages = [BASE_COVER]
    if with_mint:
        cover_pages.append(MINT_COVER)

    story = []
    for i, cover in enumerate(cover_pages):
        story.append(Spacer(1, 1.15 * inch))
        story.append(Paragraph(cover, body_style))
        if i < len(cover_pages) - 1:
            story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return buf


def render_page(pdf: PdfPages, case: ParsedCase, case_id: str, question: str,
                lookahead: int, calibrated_mint: Optional[float] = None,
                shap_png: Optional[Path] = None, with_mint: bool = False):
    t_now = case.t_now
    # Prediction horizon runs from t_now+lookahead to t_now+HORIZON (title says
    # "next {lookahead} to 60 minutes"); extend the axis to show that window.
    HORIZON = 60
    xmax = t_now + HORIZON + max(1, int(0.03 * (t_now + HORIZON)))

    # The header + eight time-aligned panels; these ratios are the blinded layout
    # and are shared by both versions so the panels render at an identical size.
    FIG_W = 16
    LEFT, RIGHT, TOP, BOTTOM, HSPACE = 0.09, 0.93, 0.93, 0.1, 0.45
    base_ratios = [1.5, 1.3, 1.3, 1.3, 1.3, 1.3, 1.5, 1.6, 1.8]
    base_h = 11  # figure height (inches) of the blinded version

    # The MINT version inserts one extra row for the SHAP force plot between the
    # header and the SpO2 panel. The SHAP images are very wide (~6:1); rendering
    # them with aspect="auto" into a short cell squishes them. Instead, size the
    # SHAP row to the image's NATIVE aspect ratio (given the fixed panel width) and
    # grow the whole figure height to fit it — so every other panel keeps exactly
    # the size it has in the blinded version.
    if with_mint:
        panel_w_in = (RIGHT - LEFT) * FIG_W
        img_aspect = 6.2  # fallback if the PNG is missing
        if shap_png is not None and Path(shap_png).exists():
            import matplotlib.image as mpimg
            _img = mpimg.imread(str(shap_png))
            img_aspect = _img.shape[1] / _img.shape[0]
        else:
            _img = None
        shap_h_in = panel_w_in / img_aspect
        # Inches-per-axes-height-unit is fixed by the blinded layout; convert the
        # desired SHAP height (inches) into a gridspec ratio and add its cell.
        n_ref = len(base_ratios)
        f_ref = (TOP - BOTTOM) / (1 + (n_ref - 1) * HSPACE / n_ref)
        in_per_unit = base_h * f_ref / sum(base_ratios)
        shap_ratio = shap_h_in / in_per_unit
        ratios = [base_ratios[0], shap_ratio] + base_ratios[1:]
        n = len(ratios)
        f_axes = (TOP - BOTTOM) / (1 + (n - 1) * HSPACE / n)
        fig_h = in_per_unit * sum(ratios) / f_axes
        fig = plt.figure(figsize=(FIG_W, fig_h))
        gs = fig.add_gridspec(n, 1, height_ratios=ratios, hspace=HSPACE,
                              left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
        ax_hdr = fig.add_subplot(gs[0])
        ax_shap = fig.add_subplot(gs[1])
        panel0 = 2
    else:
        fig = plt.figure(figsize=(FIG_W, base_h))
        gs = fig.add_gridspec(
            len(base_ratios), 1, height_ratios=base_ratios,
            hspace=HSPACE, left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM,
        )
        ax_hdr = fig.add_subplot(gs[0])
        ax_shap = None
        _img = None
        panel0 = 1
    ax_spo2 = fig.add_subplot(gs[panel0])
    ax_hr = fig.add_subplot(gs[panel0 + 1], sharex=ax_spo2)
    ax_resp = fig.add_subplot(gs[panel0 + 2], sharex=ax_spo2)
    ax_bp = fig.add_subplot(gs[panel0 + 3], sharex=ax_spo2)
    ax_tg = fig.add_subplot(gs[panel0 + 4], sharex=ax_spo2)
    ax_o2 = fig.add_subplot(gs[panel0 + 5], sharex=ax_spo2)
    ax_med = fig.add_subplot(gs[panel0 + 6], sharex=ax_spo2)
    ax_proc = fig.add_subplot(gs[panel0 + 7], sharex=ax_spo2)

    # --- Header ---
    ax_hdr.axis("off")
    ax_hdr.text(0, 1.1, f"Case {case_id}", fontsize=17, fontweight="bold",
                va="top", transform=ax_hdr.transAxes)
    ax_hdr.text(0, 0.78, question, fontsize=14, fontweight="bold", color="#333333",
                va="top", wrap=True, transform=ax_hdr.transAxes)

    # --- SHAP force plot (MINT version only) ---
    if with_mint:
        # Title the SHAP band with the MINT probability so it labels the plot and
        # never collides with the review form field in the top-right corner.
        prob = f"{calibrated_mint:.1%}" if calibrated_mint is not None else "N/A"
        ax_shap.axis("off")
        ax_shap.set_title(
            f"Probability predicted by MINT: {prob}     "
            f"MINT feature attribution (SHAP)",
            fontsize=13, loc="left", color=C_SPO2, fontweight="bold",
        )
        if _img is not None:
            # The SHAP row was sized to the image's native aspect ratio, so
            # aspect="auto" fills it exactly without distortion.
            ax_shap.imshow(_img, aspect="auto")
        else:
            ax_shap.text(0.5, 0.5, "SHAP explanation not yet available",
                         transform=ax_shap.transAxes, ha="center", va="center",
                         color="grey", fontsize=12, style="italic")
    if case.weights:
        w = case.weights
        wtxt = f"{w[0]:g} kg" if len(set(w)) == 1 else f"{min(w):g}-{max(w):g} kg"
    else:
        wtxt = "?"
    fields = [
        ("Arrival", case.arrival or "?"),
        ("Age", case.age or "?"),
        ("Sex", case.sex or "N/A"),
        ("ESI/Acuity", case.acuity or "?"),
        ("Weight", wtxt),
        ("Chief complaint", case.cc or "?"),
    ]
    line = "     ".join(f"{k}: {v}" for k, v in fields)
    ax_hdr.text(0, 0.25, line, fontsize=13, va="center", clip_on=False,
                transform=ax_hdr.transAxes)

    # --- Respiratory lines ---
    _plot_series(ax_spo2, case.series.get("SpO2"), C_SPO2, "SpO2", connect=True,
                 unit=UNITS["SpO2"])
    ax_spo2.set_ylabel("SpO2 (%)", fontsize=13)

    _plot_series(ax_hr, case.series.get("Pulse"), C_HR, "Pulse", connect=True,
                 unit=UNITS["Pulse"])
    ax_hr.set_ylabel("Heart rate\n(bpm)", fontsize=13)

    _plot_series(ax_resp, case.series.get("Resp"), C_RESP, "Resp", connect=True,
                 unit=UNITS["Resp"])
    ax_resp.set_ylabel("Resp rate\n(/min)", fontsize=13)

    # --- Blood pressure (points) ---
    # Show Systolic + Diastolic, and MAP only at timestamps that have neither
    # (a MAP measured alongside a cuff reading is redundant, so it is hidden).
    _plot_series(ax_bp, case.series.get("Systolic"), C_SYS, "Systolic", connect=False,
                 unit=UNITS["Systolic"])
    _plot_series(ax_bp, case.series.get("Diastolic"), C_DIA, "Diastolic", connect=False,
                 unit=UNITS["Diastolic"])
    sysdia_times = {t for t, _ in case.series.get("Systolic", [])}
    sysdia_times |= {t for t, _ in case.series.get("Diastolic", [])}
    map_solo = [(t, v) for t, v in case.series.get("MAP", []) if t not in sysdia_times]
    _plot_series(ax_bp, map_solo, C_MAP, "MAP", connect=False, unit=UNITS["MAP"])
    ax_bp.set_ylabel("BP (mmHg)", fontsize=13)
    if case.series.get("Systolic") or case.series.get("Diastolic") or map_solo:
        ax_bp.legend(fontsize=10, loc="upper right", ncol=3)

    # --- Temp + GCS (points, twin axis) ---
    _plot_series(ax_tg, case.series.get("Temp"), C_TEMP, "Temp", connect=False,
                 unit=UNITS["Temp"])
    ax_tg.set_ylabel("Temp (F)", fontsize=13, color=C_TEMP)
    ax_tg.tick_params(axis="y", labelcolor=C_TEMP)
    ax_gcs = ax_tg.twinx()
    _plot_series(ax_gcs, case.series.get("GCS"), C_GCS, "GCS", connect=False,
                 unit=UNITS["GCS"])
    ax_gcs.set_ylabel("GCS", fontsize=13, color=C_GCS)
    ax_gcs.tick_params(axis="y", labelcolor=C_GCS)
    ax_gcs.set_ylim(3, 15)
    ax_gcs.set_yticks([3, 15])

    # --- O2 support: device step + flow rate ---
    # Order devices on the y-axis by respiratory acuity (highest severity on top).
    if case.devices:
        present = list(dict.fromkeys(d for _, d in case.devices))
        dev_order = sorted(present, key=lambda d: DEVICE_SEVERITY.get(d, -1))
        ymap = {d: i for i, d in enumerate(dev_order)}
        dd = sorted(case.devices)
        ax_o2.step([t for t, _ in dd], [ymap[d] for _, d in dd],
                   where="post", color=C_DEVICE, linewidth=2, marker="o", markersize=6)
        ax_o2.set_yticks(range(len(dev_order)))
        ax_o2.set_yticklabels(dev_order, fontsize=9)
        ax_o2.set_ylim(-0.5, len(dev_order) - 0.5)
    else:
        ax_o2.set_yticks([])
        ax_o2.text(0.5, 0.5, "no O2 device readings", transform=ax_o2.transAxes,
                   ha="center", va="center", color="grey", fontsize=11, style="italic")
    ax_o2.set_ylabel("O2 device", fontsize=13)
    if case.flows:
        ax_flow = ax_o2.twinx()
        ff = sorted(case.flows)
        fxs = [t for t, _ in ff]
        fys = [v for _, v in ff]
        ax_flow.plot(fxs, fys, "o", color=C_FLOW, markersize=8)
        ax_flow.set_ylabel("O2 flow (L/min)", fontsize=12, color=C_FLOW)
        ax_flow.tick_params(axis="y", labelcolor=C_FLOW)
        for x, y in zip(fxs, fys):
            ax_flow.annotate(f"{y:g} L/min", (x, y), xytext=(0, 5),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=7, color=C_FLOW, clip_on=False)

    # --- Meds & Procedures/Labs ---
    _vline_track(ax_med, case.meds, C_MED, xmax)
    ax_med.set_ylabel("Medications", fontsize=13)
    _vline_track(ax_proc, case.procs, C_PROC, xmax)
    ax_proc.set_ylabel("Procedures\n& labs", fontsize=13)
    ax_proc.set_xlabel("Time since arrival (min)", fontsize=13)

    # --- "Now" line + prediction window across every panel ---
    panels = [ax_spo2, ax_hr, ax_resp, ax_bp, ax_tg, ax_o2, ax_med, ax_proc]
    for ax in panels:
        ax.set_xlim(min(0, -1), xmax)
        # zorder=0 keeps the Now line behind everything (e.g. a med/proc line at
        # exactly t_now stays visible on top of it).
        ax.axvline(t_now, color=C_NOW, linewidth=2, zorder=0)
        ax.axvspan(t_now + lookahead, t_now + HORIZON, color=C_NOW, alpha=0.15,
                   zorder=0)
        ax.grid(True, axis="x", alpha=0.2)
    ax_spo2.annotate("Now", xy=(t_now, 1.02), xycoords=("data", "axes fraction"),
                     ha="center", fontsize=12, color="black", fontweight="bold")

    # --- Leftover tokens (text at the bottom) ---
    if case.leftover:
        items = ", ".join(
            f"{n} (t={t} min)" if t is not None else n for t, n in case.leftover
        )
        fig.text(0.11, 0.015, f"Leftover tokens: {items}", fontsize=9,
                 color="dimgrey", wrap=True)

    pdf.savefig(fig)
    plt.close(fig)


# Figure-fraction position of the "Risk ratio for PPV:" prompt: top-right corner
# of the page. _FORM_Y is the baseline of the label / top of the input box.
_FORM_RIGHT = 0.98   # right edge of the input box (figure fraction)
_FORM_Y = 0.98


def _add_form_fields(src_buf: io.BytesIO, out_path: Path, n_pages: int,
                     with_mint: bool = False):
    """Overlay an interactive, fillable "Risk ratio for PPV:" text field on each
    page of the matplotlib PDF in ``src_buf`` and write the result to ``out_path``.

    Each page gets its own named field (risk_ratio_page_{i}), so a reviewer's typed
    value is independent per case and persists when they save the PDF in any viewer
    that supports AcroForms (Acrobat, Preview, Edge, etc.).

    If reportlab/pypdf are unavailable, falls back to writing the plain PDF (no
    fields) so the tool still works.
    """
    try:
        from pypdf import PdfReader, PdfWriter
        from reportlab.lib.colors import black, white
        from reportlab.pdfgen import canvas
    except ImportError:
        print("  reportlab/pypdf not installed; writing PDF without input fields. "
              "Install with: pip install reportlab pypdf")
        with open(out_path, "wb") as f:
            f.write(src_buf.getvalue())
        return

    mpl = PdfReader(src_buf)
    w = float(mpl.pages[0].mediabox.width)
    h = float(mpl.pages[0].mediabox.height)

    # Build a reportlab overlay with one page (label + text field) per case, in the
    # top-right corner. Box + text are 2x the previous height.
    from reportlab.pdfbase.pdfmetrics import stringWidth

    box_w, box_h = 190, 40
    label = "Probability of PPV initiation in the next 5-60 minutes (enter a number, 0 to 100):"
    label_font, label_size = "Helvetica-Bold", 20
    label_w = stringWidth(label, label_font, label_size)
    box_x = _FORM_RIGHT * w - box_w
    box_y = _FORM_Y * h - box_h

    ov = io.BytesIO()
    c = canvas.Canvas(ov, pagesize=(w, h))
    for i in range(n_pages):
        c.setFont(label_font, label_size)
        # Label sits to the left of the box, vertically centred on it.
        c.drawString(box_x - label_w - 12, box_y + (box_h - label_size) / 2 + 3, label)
        c.acroForm.textfield(
            name=f"risk_ratio_page_{i}", x=box_x, y=box_y, width=box_w, height=box_h,
            borderStyle="inset", borderColor=black, fillColor=white,
            textColor=black, fontSize=24, forceBorder=True,
        )
        c.showPage()
    c.save()
    ov.seek(0)

    # Clone the writer FROM the form PDF so the /AcroForm + field widgets are kept,
    # then stamp the matplotlib graphics underneath each page (over=False).
    writer = PdfWriter(clone_from=ov)
    for i, page in enumerate(writer.pages):
        page.merge_page(mpl.pages[i], over=False)
    cover_buf = _build_cover_pdf((w, h), with_mint=with_mint)
    cover_reader = PdfReader(cover_buf)
    for i, page in enumerate(cover_reader.pages):
        writer.insert_page(page, i)
    writer.set_need_appearances_writer(True)
    with open(out_path, "wb") as f:
        writer.write(f)


def _render_cohort_pdf(df, question, lookahead, out_path, shap_dir, with_mint,
                       max_pages=None):
    """Render one cohort's pages to a PDF (blinded, or the MINT-reveal version)."""
    rows = df.itertuples(index=False)
    n = len(df) if max_pages is None else min(max_pages, len(df))
    print(f"Rendering {n} pages -> {out_path}")
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for i, r in enumerate(rows):
            if max_pages is not None and i >= max_pages:
                break
            case = ParsedCase(str(r.trajectory))
            shap_png = Path(shap_dir) / f"{r.case_id}.png" if with_mint else None
            cal = getattr(r, "calibrated_mint", None) if with_mint else None
            render_page(pdf, case, case_id=r.case_id, question=question,
                        lookahead=lookahead, calibrated_mint=cal,
                        shap_png=shap_png, with_mint=with_mint)
    buf.seek(0)
    _add_form_fields(buf, out_path, n_pages=n, with_mint=with_mint)
    print(f"Saved {out_path}")


def main(
    excel: str,
    out_dir: Optional[str] = None,
    mode: Optional[str] = None,
    lookahead: int = 5,
    max_pages: Optional[int] = None,
):
    """
    Render each subgroup (A/B/C/D) to two PDFs: a blinded review version and a
    MINT-reveal version that adds the SHAP force plot and MINT probability.

    Produces eight files: Cohort_{A,B,C,D}.pdf (blinded) and
    Cohort_{A,B,C,D}_With_MINT.pdf (adds the SHAP band + MINT probability). All
    eight carry the fillable per-page review field for the human reviewer.

    Args:
        excel: Path to the workbook produced by fig1_excel.py.
        out_dir: Output directory (default: next to the workbook).
        mode: Task mode for the prediction-question title (default: inferred from
              the Excel filename stem).
        lookahead: Prediction-window length in minutes (must match fig1_excel.py).
        max_pages: Render only the first N pages per cohort (quick preview).
    """
    excel_path = Path(excel)
    # The "model" sheet carries subgroup + calibrated_mint (the MINT-reveal
    # version needs both); the trajectory column is identical to "human".
    df = pd.read_excel(excel_path, sheet_name="model")

    mode = mode or excel_path.stem
    question = QUESTIONS.get(mode, "What happens in the next {la} to 60 minutes?").format(la=lookahead)

    out_dir = Path(out_dir) if out_dir else excel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_dir = excel_path.parents[1] / "shap"

    for cohort in ["A", "B", "C", "D"]:
        sub = df[df["subgroup"] == cohort].reset_index(drop=True)
        if sub.empty:
            print(f"Cohort {cohort}: no cases, skipping")
            continue
        _render_cohort_pdf(sub, question, lookahead, out_dir / f"Cohort_{cohort}.pdf",
                           shap_dir, with_mint=False, max_pages=max_pages)
        _render_cohort_pdf(sub, question, lookahead,
                           out_dir / f"Cohort_{cohort}_With_MINT.pdf",
                           shap_dir, with_mint=True, max_pages=max_pages)


if __name__ == "__main__":
    tapify(main)
