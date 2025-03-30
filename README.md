# Model-Tournament-Theory
Superstar Effects in ATP Tennis Tournaments

This repository contains the code, data, and analysis files used in the research project titled "Super-Star Effects in ATP Tennis Tournaments". The project investigates how the presence of top-ranked tennis players—Novak Djokovic, Rafael Nadal, and Roger Federer—affects the performance of other players during ATP tournaments.
📘 Project Origin

This research originated from an idea proposed by Sebastian Bervoets (Aix-Marseille School of Economics - AMSE) during my Master 1 internship in 2020. It was further developed as part of an academic research project combining sports economics, tournament theory, and causal inference.
🧠 Project Overview

The goal is to quantify the influence of tennis "superstars" on the behavior and outcomes of other competitors using a mix of statistical and econometric methods, including:

    Panel models with fixed and random effects

    Survival models

    Logit and Hurdle models

    Propensity score matching (causal inference)

The project covers more than 900,000 ATP matches (1968–2023) and focuses on 14,710 cleaned matches in selected high-level tournaments (Grand Slams, Masters 1000, ATP 250/500).
📂 Repository Structure

    ├── Theorie_Tournois_Rapport_Cresus.pdf    # Final PDF report (academic article style)
    ├── process_models.ipynb                   # Python notebook for processing data and running models
    ├── modules/                               # Custom Python modules (data cleaning, modeling, visualization)
    │   ├── data_processor.py
    │   ├── ranking_star.py
    │   └── ... (more module)
    ├── output/                                # Results, tables, and figures
    │   ├── match_descript.png
    │   ├── survival_all.png
    │   └── ... (more plots and tables)

🔍 Key Features

    Tournament Theory Applied to Tennis: Quantifying effort, performance, and competition intensity based on the presence of superstars.

    Star Identification Algorithm: A dynamic method for detecting dominant players based on ranking and cumulative points.

    Causal Inference via Matching: Estimating the impact of star presence on:

        Match outcomes for lower-ranked players

        Progression in tournaments

        Number of sets won

    Survival and Panel Models: Evaluating how star density affects survival in tournaments and dominant wins.

📊 Methods and Models

    Survival Analysis (Kaplan-Meier, Log-rank tests)

    Panel Regression (Fixed Effects, Random Effects)

    Logistic and Hurdle Models

    Propensity Score Matching (treatment = number of stars)

📈 Main Results

    The presence of stars significantly reduces the success rate of lower-ranked players.

    Tournaments with high star density show increased dominance and reduced effort from outsiders.

    The effect varies by superstar: Djokovic tends to amplify dominance, while Federer and Nadal have more nuanced effects.

    Causal models confirm the deterrent impact of superstars on competition dynamics.

📎 How to Use This Repository

    Clone the repository:

    git clone https://github.com/yourusername/superstar-effects-atp.git
    cd superstar-effects-atp

    Install the required packages (listed in notebook or module headers).

    Open and run process_models.ipynb to reproduce core analyses and figures.

📚 Reference Report

The full detailed report is available here:
📄 Theorie_Tournois_Rapport_Cresus.pdf

It includes:

    Context and literature

    Data cleaning and variable construction

    Modeling details

    Full results and interpretations

    Figures and tables

👨‍🏫 Credits

Crésus Kounoudji
Research conducted under the supervision of Sebastian Bervoets (AMSE)
Initial idea proposed during an internship at Aix-Marseille School of Economics in 2020.
📌 Next Steps and Extensions

    Add simulations or dynamic models of competition under counterfactual reward schemes.

    Extend the matching strategy with DID or synthetic control.

    Publish a shorter version in an academic journal.
