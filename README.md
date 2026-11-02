# 📊 Credit Portfolio Segmentation and Risk Pattern Analysis

An unsupervised machine-learning project that analyzes **1,346,632 Lending Club loan records** to identify borrower segments and understand historical repayment patterns.

* **Clustering** – identify five borrower segments using K-Means, with Hierarchical Clustering and DBSCAN for comparison.
* **Risk Analysis** – compare historical default rates across borrower groups without using default information during clustering.
* **Dashboard** – an interactive Streamlit application for exploring borrower segments, repayment patterns and individual borrower classification into a segment.


## Getting Started

**1. Install the dependencies** (Python 3.10 or later is recommended).

```bash
pip install -r requirements.txt
```

**2. Execute the notebook** (execution time depends on your system).

```bash
jupyter notebook Credit_Portfolio_Segmentation.ipynb
```

Select **Kernel → Restart & Run All** and wait until all cells finish executing. The notebook creates `models/model.pkl` and the supporting CSV files inside `data/processed/`.

**3. Launch the Streamlit dashboard**

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser to access the application.

## Notebook Workflow

| Step | Implementation |
|---|---|
| 1 | Import the Lending Club dataset containing approximately 1.35 million loan records |
| 2 | Remove duplicates, clean categorical values, handle missing values and filter invalid DTI and home ownership records |
| 3 | Generate financial features such as log income, loan-to-income ratio and employment length |
| 4 | Perform EDA using distributions, categorical comparisons, default rates, correlation analysis and yearly trends |
| 5 | Apply StandardScaler and One-Hot Encoding to prepare numerical and categorical features |
| 6 | Apply PCA to reduce dimensionality while retaining 94% of the total variance using seven components |
| 7 | Compare K-Means, Hierarchical Clustering and DBSCAN using a 10,000-borrower sample |
| 8 | Train the final K-Means model with five clusters using the complete dataset |
| 9 | Visualize borrower groups using PCA, t-SNE and UMAP, followed by cluster profiling |
| 10 | Analyze historical default patterns, detect anomalies using DBSCAN and export the trained model using pickle |

### Preventing Data Leakage

* The `Default` column is excluded from clustering features and used only for post-clustering analysis.
* Loan IDs, descriptions, titles and other unsuitable columns are excluded from the clustering inputs.
* Issue dates are used for yearly trend analysis, not for clustering.
* Numerical features are standardized so that larger financial values do not dominate the clustering process.
* PCA is applied to the prepared feature matrix before clustering.
* Hierarchical Clustering and DBSCAN are evaluated on a random sample of 10,000 borrowers because of their computational requirements.
* The final K-Means model is trained using the complete cleaned dataset.

## Model Evaluation

The following metrics were obtained from the 10,000-borrower sample used for algorithm comparison.

**Clustering – selected model: K-Means (k = 5)**

| Algorithm | Clusters | Noise (%) | Silhouette | Davies-Bouldin |
|---|---:|---:|---:|---:|
| K-Means | 5 | 0.00 | 0.145 | 1.701 |
| Hierarchical (Ward) | 5 | 0.00 | 0.068 | 2.311 |
| DBSCAN | 1 | 2.19 | N/A | N/A |

K-Means was selected because it produced more interpretable borrower segments and better clustering metrics than Hierarchical Clustering. DBSCAN identified one main dense group rather than multiple useful segments.

## Key Observations

* K-Means identified five borrower segments: Established Homeowners, Early-Career Renters, Stretched Low-Income, Large-Loan High-Income and Prime Credit.
* The overall portfolio default rate is approximately 20.0%, with segment default rates ranging from 8.38% to 29.08%.
* Stretched Low-Income borrowers have the highest default rate of 29.08%, approximately 1.46 times the portfolio average.
* Prime Credit borrowers have the lowest default rate of 8.38%, approximately 0.42 times the portfolio average.
* FICO score alone does not explain repayment behaviour. Affordability measures such as DTI and loan-to-income ratios also provide useful insights.
* DBSCAN identifies 219 unusual borrowers (2.19% of the sample), mainly associated with unusually high FICO scores or other financial characteristics. Their default rate is not higher than that of normal borrowers.

## Dashboard Features

* **📌 Overview** – displays total borrowers, overall default rate, number of segments and Silhouette Score. Includes borrower distribution, default rate comparisons and algorithm performance.
* **👥 Segments** – explores individual borrower groups through income, FICO, loan amount, DTI, employment and home ownership statistics. Includes interactive segment comparisons and feature distributions.
* **📉 Default Rates** – compares historical default rates across segments with confidence intervals and examines year-wise repayment patterns.
* **🗺️ Cluster Map** – visualizes borrower groups using PCA, t-SNE and UMAP. Users can switch between segment colours and historical repayment outcomes.
* **🧑‍💼 Try a Borrower** – accepts borrower details and identifies the closest existing segment using the saved preprocessing pipeline, PCA and K-Means model.
* **💡 Insights** – presents key findings, possible banking applications and project limitations.

The dashboard uses a green-and-white theme, a dark sidebar, interactive metric cards, dropdown selections and charts for exploring borrower patterns.

## Troubleshooting

* **`model.pkl` not found** – execute the notebook successfully and ensure the generated model is stored inside the `models` folder.
* **Dataset missing** – place `loans_granting_model_dataset.csv` in the notebook's expected input directory.
* **Processed CSV files missing** – execute the notebook's export section to generate the files inside `data/processed/`.
* **Pickle compatibility problems** – load the saved model using compatible library versions; rerun the notebook if necessary.
* **Missing Python packages** – install the dependencies listed in `requirements.txt`.
* **Slow execution** – PCA, full-dataset K-Means and dimensionality reduction can take longer depending on available memory and processing power.

## Limitations

* The Silhouette Score of 0.145 indicates considerable overlap between borrower segments. They represent general financial patterns rather than sharply separated groups.
* DBSCAN identifies one main dense area and a small number of noise points, so it is not suitable for creating the five borrower segments in this dataset.
* The dashboard's individual borrower tool identifies the closest existing segment; it does not predict an individual's probability of default.
* Historical default rates describe past borrower groups and should not be treated as guaranteed future outcomes.
* Recent loans from 2017–2018 have shorter repayment histories, which may affect their observed default rates.
* The analysis is based on historical Lending Club data and may not represent current lending conditions.
