# Multilingual Social Discourse Analysis

This project implements a comprehensive Natural Language Processing (NLP) pipeline for analyzing social discourse across multiple languages. It uses the **XLM-RoBERTa** model to perform three distinct classification tasks related to polarization and discourse dynamics.

## Project Structure

The project is divided into three main subtasks, each implemented in a separate Jupyter Notebook:

### 1. Subtask 1: Binary Classification (Polarization)
**File:** `NLP_Subtask_1.ipynb`

*   **Goal:** Classify text as either **Polarized** or **Non-Polarized**.
*   **Model:** XLM-RoBERTa (base).
*   **Key Features:**
    *   **Data Augmentation:** Uses NLLB-200 to translate English samples to Low-Resource Languages (e.g. Amharic) to address data scarcity.
    *   **Rigorous Splitting:** Implements a strict train/validation/held-out test split (20% held-out).
    *   **Class Balancing:** Applies Random Oversampling to handle class imbalance.

### 2. Subtask 2: Polarization Type Classification
**File:** `NLP_SUBTASK2_FINAL_(1).ipynb`

*   **Goal:** Classify the *type* of polarization present in the text into 5 categories:
    1.  Political/Ideological
    2.  Racial/Ethnic
    3.  Religious
    4.  Gender/Sexual
    5.  Other
*   **Methodology:**
    *   **Experiment 1 (Robust Baseline):** Uses Focal Loss with Inverse-Frequency Class Weights.
    *   **Experiment 2 (Proposal-Aligned "Turbo"):** Implements **Supervised Contrastive Learning (SCL)** with a projection head for joint optimization (Loss = Focal + 0.5 * SCL) and Dynamic Thresholding.
    *   **Experiment 3 (Scale Comparison):** Benchmarks Base vs. Large models.

### 3. Subtask 3: Multi-Label Classification (Discourse Features)
**File:** `NLP_Subtask_3.ipynb`

*   **Goal:** Identify specific discourse features (Multi-Label classification) such as:
    *   Stereotype
    *   Vilification
    *   Dehumanization
    *   Extreme Language
    *   Lack of Empathy
    *   Invalidation
*   **Experiments:**
    *   Compares **Baseline Classification** (Standard XLM-R) vs. **Contrastive Joint Learning** (Custom Architecture with SupCon Loss).
    *   Evaluates performance on both Standard Multilingual Data and Oversampled Data.

## Installation & Requirements

To run these notebooks, you will need a Python environment with the following libraries:

```bash
pip install torch transformers pandas numpy scikit-learn accelerate sentencepiece tqdm
```

*   **Python:** 3.8+
*   **PyTorch:** Compatible with your CUDA version (for GPU acceleration).
*   **Transformers:** Hugging Face Transformers library.

## Dataset

The dataset used for this project identifies polarization and social discourse features across multiple languages (English, Spanish, Arabic, Amharic, Chinese, Urdu, etc.).

[link](https://drive.google.com/drive/folders/1YD5wkRlzLJbm_qqAlZlTCdbPVZKMO7SQ?usp=sharing)

*   **Data Format:** The notebooks expect CSV files organized by task and language (e.g., `subtask1/train/eng.csv`).
*   **Setup:** The notebooks include code to handle data loading, often expecting a zip file or a specific directory structure in Google Drive (`/content/drive/MyDrive/NLP-PROJECT`).

## Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    ```
2.  **Dataset:** Ensure you have the dataset downloaded and the path configured correctly in the notebook (or upload the zip file if running on Colab).
3.  **Run Notebooks:** Open the desired subtask notebook (e.g., `NLP_Subtask_1.ipynb`) in Jupyter Lab, Jupyter Notebook, or Google Colab and execute the cells sequentially.

## Acknowledgements

*   Model: [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base)
*   Translation: [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
