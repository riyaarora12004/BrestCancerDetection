import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = {
    "mean radius":          ("Cell size",          "Average size of the cell nuclei. Larger cells can be a warning sign."),
    "mean texture":         ("Cell graininess",    "How uneven / grainy the cells look under the microscope."),
    "mean area":            ("Cell area",          "Average surface area of the cells."),
    "mean smoothness":      ("Edge smoothness",    "How regular the cell edges are. Rough, irregular edges raise concern."),
    "mean concavity":       ("Dent depth",         "How deep the dents are in the cell outline."),
    "mean concave points":  ("Number of dents",    "How many dents / notches the cell outline has."),
}
COLS = list(FEATURES)

# ---------------------------------------------------------------
# 2. Load data and train the two models
# ---------------------------------------------------------------
def train():
    d = load_breast_cancer(as_frame=True)
    X, y = d.data[COLS], d.target            # y: 0 = malignant, 1 = benign
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=7),
    }
    stats = {}
    for name, m in models.items():
        cv = cross_val_score(m, Xtr, ytr, cv=5).mean()
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        caught, missed, false_alarm, ok_benign = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        stats[name] = dict(acc=accuracy_score(yte, pred), cv=cv,
                           caught=int(caught), missed=int(missed), false_alarm=int(false_alarm),
                           total_cancer=int(caught + missed), total_benign=int(false_alarm + ok_benign),
                           n=len(yte))
    return d, X, y, models, stats


def risk_of_cancer(model, row):
    """Probability (0-1) that the sample is malignant."""
    return float(model.predict_proba(row)[0][0])     


def explain(lr_model, row):
    """How much each input pushes the risk UP (+) or DOWN (-) in Logistic Regression."""
    scaler, lr = lr_model.steps[0][1], lr_model.steps[1][1]
    z = scaler.transform(row)[0]
    return -lr.coef_[0] * z                           

BG, BLUE = "#f5f7fb", "#2b5fd9"


class App:
    def __init__(self, root):
        self.root = root
        root.title("Breast Cancer Risk Screening Helper")
        root.geometry("1150x740")
        root.configure(bg=BG)
        self.data, self.X, self.y, self.models, self.stats = train()
        self.vars, self.val_lbls = {}, {}

        hdr = tk.Frame(root, bg=BLUE)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Breast Cancer Risk Screening Helper", bg=BLUE, fg="white",
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=16, pady=10)
        tk.Label(hdr, text="Educational demo - not a medical diagnosis", bg=BLUE, fg="#ffd9d9",
                 font=("Segoe UI", 10, "italic")).pack(side="right", padx=16)

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        self.t_check, self.t_about = tk.Frame(nb, bg=BG), tk.Frame(nb, bg=BG)
        nb.add(self.t_check, text="  Risk Check  ")
        nb.add(self.t_about, text="  How it works & accuracy  ")
        self.build_check()
        self.build_about()
        self.preset("benign")

    # ----- Tab 1 -----
    def build_check(self):
        left = tk.LabelFrame(self.t_check, text=" Step 1: Enter the biopsy measurements ", bg=BG,
                             font=("Segoe UI", 10, "bold"))
        left.pack(side="left", fill="y", padx=8, pady=8)
        for col, (nice, desc) in FEATURES.items():
            lo, hi, med = self.X[col].min(), self.X[col].max(), self.X[col].median()
            frm = tk.Frame(left, bg=BG)
            frm.pack(fill="x", padx=10, pady=6)
            tk.Label(frm, text=nice, bg=BG, font=("Segoe UI", 10, "bold")).pack(anchor="w")
            tk.Label(frm, text=desc, bg=BG, fg="#555", font=("Segoe UI", 8), wraplength=340,
                     justify="left").pack(anchor="w")
            row = tk.Frame(frm, bg=BG)
            row.pack(fill="x")
            v = tk.DoubleVar(value=med)
            s = ttk.Scale(row, from_=lo, to=hi, variable=v, length=290,
                          command=lambda _e, c=col: self.update())
            s.pack(side="left")
            lbl = tk.Label(row, width=7, bg=BG, font=("Consolas", 10, "bold"))
            lbl.pack(side="left", padx=6)
            self.vars[col], self.val_lbls[col] = v, lbl
        b = tk.Frame(left, bg=BG)
        b.pack(pady=10)
        ttk.Button(b, text="Example: typical harmless case", command=lambda: self.preset("benign")).pack(side="left", padx=4)
        ttk.Button(b, text="Example: typical cancer case", command=lambda: self.preset("malignant")).pack(side="left", padx=4)

        right = tk.LabelFrame(self.t_check, text=" Step 2: Result ", bg=BG, font=("Segoe UI", 10, "bold"))
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.verdict = tk.Label(right, text="", font=("Segoe UI", 20, "bold"), bg=BG)
        self.verdict.pack(pady=(10, 0))
        self.risk_txt = tk.Label(right, text="", font=("Segoe UI", 11), bg=BG, justify="center")
        self.risk_txt.pack()
        self.bar = tk.Canvas(right, width=520, height=34, bg="white", highlightthickness=1,
                             highlightbackground="#bbb")
        self.bar.pack(pady=8)
        self.models_txt = tk.Label(right, text="", font=("Segoe UI", 10), bg=BG, justify="left")
        self.models_txt.pack()
        self.advice = tk.Label(right, text="", font=("Segoe UI", 10), bg="#fff7e0", wraplength=560,
                               justify="left", padx=10, pady=8)
        self.advice.pack(fill="x", padx=14, pady=8)
        tk.Label(right, text="What is pushing the risk up or down?", bg=BG,
                 font=("Segoe UI", 10, "bold")).pack()
        self.fig = plt.Figure(figsize=(5.8, 2.6), dpi=100)
        self.fig.patch.set_facecolor(BG)
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)

    def current_row(self):
        import pandas as pd
        return pd.DataFrame([[self.vars[c].get() for c in COLS]], columns=COLS)

    def preset(self, kind):
        label = 1 if kind == "benign" else 0
        med = self.X[self.y == label].median()
        for c in COLS:
            self.vars[c].set(med[c])
        self.update()

    def update(self):
        row = self.current_row()
        for c in COLS:
            self.val_lbls[c].config(text=f"{self.vars[c].get():.3g}")
        p_lr = risk_of_cancer(self.models["Logistic Regression"], row)
        p_rf = risk_of_cancer(self.models["Random Forest"], row)
        p = (p_lr + p_rf) / 2

        if p < 0.20:
            txt, col, adv = "LOW RISK", "#1a9a4a", ("These measurements look like those of harmless (benign) lumps. "
                                                    "Keep up regular check-ups.")
        elif p < 0.60:
            txt, col, adv = "UNCERTAIN", "#d98a00", ("The measurements are borderline and the models disagree or are unsure. "
                                                     "More tests are needed - please consult a doctor.")
        else:
            txt, col, adv = "HIGH RISK", "#d12b2b", ("These measurements look like those of cancerous (malignant) cells. "
                                                     "Please see a doctor soon for proper tests. "
                                                     "This is NOT a diagnosis - many worrying results turn out fine.")
        self.verdict.config(text=txt, fg=col)
        self.risk_txt.config(text=f"Combined chance of cancer: {p:.0%}")
        self.models_txt.config(text=f"Logistic Regression says {p_lr:.0%}      |      Random Forest says {p_rf:.0%}")
        self.advice.config(text=adv)
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, 520 * p, 34, fill=col, width=0)
        for x in (0.2, 0.6):
            self.bar.create_line(520 * x, 0, 520 * x, 34, fill="#555", dash=(3, 2))

        contrib = explain(self.models["Logistic Regression"], row)
        order = np.argsort(contrib)
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.barh([FEATURES[COLS[i]][0] for i in order], contrib[order],
                color=["#d12b2b" if contrib[i] > 0 else "#1a9a4a" for i in order])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("<- lowers risk          raises risk ->", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_facecolor(BG)
        self.fig.tight_layout()
        self.canvas.draw()
        
    def build_about(self):
        txt = tk.Text(self.t_about, wrap="word", font=("Segoe UI", 11), bg="white", padx=18, pady=14,
                      relief="flat")
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.tag_configure("h", font=("Segoe UI", 13, "bold"), foreground=BLUE, spacing1=10, spacing3=4)
        txt.tag_configure("b", font=("Segoe UI", 11, "bold"))

        def add(s, tag=None):
            txt.insert("end", s, tag)

        add("What does this app do?\n", "h")
        add("When a lump is found, doctors take a tiny sample (a biopsy) and look at the cells under a "
            "microscope. A computer measures things like cell size, shape and edge roughness. "
            "This app learned, from 569 real past patients, how those measurements relate to whether the lump "
            "turned out to be harmless (benign) or cancer (malignant), and uses that to estimate a risk for a new case.\n")

        add("\nThe two 'brains' (models)\n", "h")
        add("1. Logistic Regression ", "b")
        add("- a simple, transparent method. It gives each measurement a weight and adds them up into a "
            "probability. This is the model that powers the red/green 'what is pushing the risk' chart.\n")
        add("2. Random Forest ", "b")
        add("- a team of 200 'decision trees' (like yes/no flowcharts) that vote. It can catch more complex "
            "patterns. The final result shown is the average of both models, so one model's mistake is softened.\n")

        add("\nHow good are they? (tested on patients the models had never seen)\n", "h")
        for name, s in self.stats.items():
            add(f"{name}\n", "b")
            add(f"  - Correct on {s['acc']:.0%} of {s['n']} test patients "
                f"(5-fold cross-validated accuracy on training data: {s['cv']:.0%}).\n")
            add(f"  - Found {s['caught']} of {s['total_cancer']} real cancers; "
                f"missed {s['missed']}.\n")
            add(f"  - Raised {s['false_alarm']} false alarm(s) among {s['total_benign']} harmless cases.\n")

        add("\nWhat the numbers mean\n", "h")
        add("A missed cancer is the most dangerous error, so 'cancers found' matters more than plain accuracy. "
            "A false alarm just leads to extra tests - stressful, but safe.\n")

        add("\nImportant limits - please read\n", "h")
        add("- This is a student ML project, NOT a medical device.\n"
            "- The inputs are lab measurements from a biopsy image; you can't measure them at home.\n"
            "- The data comes from one hospital study; it may not work equally well for everyone.\n"
            "- Never use this to decide whether to see a doctor. If you notice a lump or any change, get it checked.\n")

        add("\nSyllabus topics used\n", "h")
        add("Data preparation (feature selection, standardization) - train/test split - Logistic Regression - "
            "Random Forest (ensemble) - cross-validation - accuracy & confusion matrix - recall - model "
            "explainability.\n")
        txt.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop() 