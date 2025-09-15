# Predicting Market Regimes with a Hidden Markov Model-Gated Mixture of Experts

This project develops a sophisticated machine learning system to identify and adapt to changing financial market regimes. The core of the model is a **Mixture of Experts (MoE)** architecture, where specialist neural networks are "gated" by a **Hidden Markov Model (HMM)** to predict stock market returns.

---

## 📊 Example Output

**Figure 1:** HMM identifying three distinct market regimes (Calm/Bullish, High-Volatility/Crash, Neutral/Choppy) on the historical S&P 500 price chart.

---

## 🔎 Overview

Financial markets are not monolithic; their behavior shifts between distinct states or **regimes**.  
A single predictive model often fails because it must average its strategy across fundamentally different environments like a stable bull market and a volatile crash.

This project implements a more adaptive system:

- **The Manager (HMM):** Learns hidden market states (unsupervised) using volatility, momentum, and other features.
- **The Specialists (Neural Experts):** A Mixture of Experts, where each expert network specializes in predicting returns within a given regime.
- **The Gate (HMM Posteriors):** The HMM provides regime probabilities, which weight each expert’s prediction for robust, context-aware forecasting.

---

## 🏗️ Project Structure

The workflow is modularized into **five Jupyter Notebooks**, designed to be run in sequential order:

1. **01_Data_Acquisition_and_Feature_Engineering.ipynb**
   - Downloads historical S&P 500 data.
   - Engineers key features (returns, rolling volatility, momentum, intraday range).
   - Saves dataset → `engineered_features.csv`.

2. **02_Regime_Identification_with_HMMs.ipynb**
   - Loads engineered features.
   - Uses **BIC** to select optimal number of regimes.
   - Trains HMM to identify regimes.
   - Saves labeled dataset → `features_with_regimes.csv`.

3. **03_Building_and_Training_the_Mixture_of_Experts.ipynb**
   - Loads regime-labeled data.
   - Defines PyTorch-based expert networks.
   - Trains Mixture of Experts with HMM probabilities as the gating mechanism.
   - Saves trained model weights.

4. **04_Evaluation_and_Strategy_Backtesting.ipynb**
   - Loads expert models.
   - Evaluates per-regime performance.
   - Implements and backtests a regime-aware trading strategy.
   - Reports Sharpe Ratio & Annualized Return vs. buy-and-hold.

5. **05_Advanced_Visualizations_and_Model_Interpretation.ipynb**
   - t-SNE visualization of regime separation.
   - Transition matrix analysis.
   - Regime-level return distribution comparisons.
   - Optional: **SHAP** analysis for expert interpretability.

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
