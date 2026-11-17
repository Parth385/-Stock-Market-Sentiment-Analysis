# Stage 3 — Model & Tokenizer Configuration Documentation

**Project**: *Beyond Sentiment: A Comparative and Temporal Analysis of Financial News Sentiment for Short-Term Indian Stock Market Prediction*  
**Research Cohort**: 15 Continuous NSE Nifty 50 Equities (2020–2026)  
**Date**: November 2026  
**Document**: Technical Configuration Specification for Linguistic & Financial Sentiment Representations  

---

## 1. Executive Overview

Stage 3 extracts financial sentiment from the verified news dataset (`data/processed/news_clean.parquet`, containing 155,537 direct article records with 68,623 unique headline strings). To rigorously investigate whether domain-specific representations offer predictive superiority over lexicon-based approaches, three distinct sentiment extraction paradigms are implemented:

1. **Loughran–McDonald (LM) Financial Dictionary** (Domain-specific financial lexicon).
2. **ProsusAI FinBERT** (Domain-adapted BERT transformer fine-tuned on Financial PhraseBank).
3. **Financial RoBERTa** (`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`, Compact, high-throughput transformer).

---

## 2. Approach A: Loughran–McDonald Financial Lexicon

### 2.1 Theoretical Foundation & Rationale
Standard sentiment lexicons (e.g., General Inquirer, Harvard IV-4) misclassify common financial words: for instance, words like *"cost"*, *"liability"*, *"tax"*, or *"expense"* are flagged as negative despite being ordinary operating terms in financial disclosures. The Loughran & McDonald (2011) master financial dictionary was constructed specifically to address this bias in corporate filings and financial press releases.

### 2.2 Dictionary Source & Composition
- **Source**: University of Notre Dame Software Repository for Accounting and Finance (SRAF), Prof. Tim Loughran & Bill McDonald.
- **Master File**: `data/raw/sentiment/loughran_mcdonald_master_dictionary.csv` (86,553 English entries).
- **Categories Utilized**:
  - **Negative Words**: 2,345 terms (e.g., *loss, default, deficit, drop, decline, plunge, fraud, litigation*).
  - **Positive Words**: 347 terms (e.g., *profit, growth, gain, surge, outperform, record, lucrative*).
  - **Uncertainty Words**: 297 terms (e.g., *approximate, risk, volatility, fluctuate, pending, anticipate*).
  - **Litigious Words**: 903 terms (e.g., *lawsuit, injunction, court, claimant, settlement, breach*).
  - **Constraining Words**: 184 terms (e.g., *mandatory, commit, strict, limit, cap, obligation*).

### 2.3 Mathematical Formulae
For each article $i$ with total word count $W_i$:
$$\text{lm\_polarity}_i = \frac{N_{\text{pos}, i} - N_{\text{neg}, i}}{N_{\text{pos}, i} + N_{\text{neg}, i} + \epsilon}, \quad \epsilon = 10^{-5}$$
$$\text{lm\_subjectivity}_i = \frac{N_{\text{pos}, i} + N_{\text{neg}, i}}{W_i + \epsilon}$$
$$\text{lm\_uncertainty}_i = \frac{N_{\text{unc}, i}}{W_i + \epsilon}$$

---

## 3. Approach B: ProsusAI FinBERT

### 3.1 Model Checkpoint & Architecture
- **Hugging Face Model Hub ID**: [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert)
- **Base Architecture**: `BertForSequenceClassification` (`bert-base-uncased`, 12 layers, 768 hidden dimensions, 12 attention heads, 110M parameters).
- **Pretraining**: Pretrained on Reuters Financial News and SEC corporate filings; fine-tuned on the Financial PhraseBank dataset (Malo et al., 2014) consisting of 4,845 financial sentences annotated by 16 financial researchers.

### 3.2 Tokenizer & Sequence Configuration
- **Tokenizer**: `BertTokenizerFast` (`AutoTokenizer.from_pretrained("ProsusAI/finbert")`).
- **Vocabulary Size**: 30,522 WordPieces.
- **Max Sequence Length**: 64 tokens.
- **Truncation Audit**: 
  - Unique headline word count: Mean = 13.2 words, Max = 41 words.
  - Across all 68,623 unique headlines, the maximum token length is **61 tokens**.
  - **Truncation Rate**: **0.00%** (zero truncation). Every news headline is evaluated in its entirety without loss of linguistic content.

### 3.3 Label Mapping & Inference Logic
- **Architecture Label Map (`id2label`)**:
  - `0`: `positive`
  - `1`: `negative`
  - `2`: `neutral`
- **Softmax Normalization**:
  $$P_k = \frac{\exp(z_k)}{\sum_{j=0}^2 \exp(z_j)}, \quad k \in \{\text{pos}, \text{neg}, \text{neu}\}$$
- **Derived Metrics**:
  - $\text{finbert\_polarity} = P_{\text{pos}} - P_{\text{neg}} \in [-1.0, 1.0]$
  - $\text{finbert\_confidence} = \max(P_{\text{pos}}, P_{\text{neg}}, P_{\text{neu}}) \in [1/3, 1.0]$

---

## 4. Approach C: Financial RoBERTa

### 4.1 Model Checkpoint & Architecture
- **Hugging Face Model Hub ID**: [`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
- **Base Architecture**: `RobertaForSequenceClassification` (`distilroberta-base`, 6 layers, 768 hidden dimensions, 12 attention heads, 82M parameters).
- **Fine-Tuning**: Fine-tuned on financial news sentiment analysis with byte-pair encoding (BPE).

### 4.2 Tokenizer & Sequence Configuration
- **Tokenizer**: `RobertaTokenizerFast` (`AutoTokenizer.from_pretrained(...)`).
- **Vocabulary Size**: 50,265 BPE tokens.
- **Max Sequence Length**: 64 tokens.
- **Truncation Audit**: Maximum token length observed across headlines is 38 tokens. Truncation rate is **0.00%**.

### 4.3 Label Mapping & Inference Logic
- **Architecture Label Map (`id2label`)**:
  - `0`: `negative`
  - `1`: `neutral`
  - `2`: `positive`
- **Softmax Normalization**:
  $$P_k = \frac{\exp(z_k)}{\sum_{j=0}^2 \exp(z_j)}, \quad k \in \{\text{pos}, \text{neg}, \text{neu}\}$$
- **Derived Metrics**:
  - $\text{roberta\_polarity} = P_{\text{pos}} - P_{\text{neg}} \in [-1.0, 1.0]$
  - $\text{roberta\_confidence} = \max(P_{\text{pos}}, P_{\text{neg}}, P_{\text{neu}}) \in [1/3, 1.0]$

---

## 5. Summary Comparison Matrix

| Specification | Loughran–McDonald | ProsusAI FinBERT | Financial RoBERTa |
| :--- | :--- | :--- | :--- |
| **Model Type** | Lexicon / Dictionary | Deep Transformer (BERT-base) | Distilled Transformer (RoBERTa) |
| **Checkpoint** | ND-SRAF Master Dict 2023 | `ProsusAI/finbert` | `mrm8488/distilroberta-...` |
| **Parameters** | None (Static Counts) | 110,000,000 | 82,000,000 |
| **Context Window** | Whole Text / Headline | 64 tokens (untruncated) | 64 tokens (untruncated) |
| **Output Classes** | Word Count Densities | 3 (Pos, Neu, Neg) | 3 (Neg, Neu, Pos) |
| **Handling of Negation** | Basic distance windowing | Full bidirectional attention | Full bidirectional attention |
| **Speed (CPU Throughput)** | > 5,000 texts/sec | ~60–70 texts/sec | ~120–150 texts/sec |
| **Storage Output** | `sentiment_lm.parquet` | `sentiment_finbert.parquet` | `sentiment_roberta.parquet` |
