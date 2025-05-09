![Project](https://img.shields.io/badge/Project-blacksmithing-blue)
![Author](https://img.shields.io/badge/Author-aquinordg-green)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Version](https://img.shields.io/badge/Version-1.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

# ⚒️ Blacksmithing ⚒️

A Python framework for semantic clustering and evaluation of binary/categorical data.

---

## 🔥 Key Features

### 🧰️ Core Framework
- **One-Command Setup**: Single installation gets you started
- **Comprehensive Documentation**: Tutorials, API reference, and practical examples
- **scikit-learn Compatible**: Familiar interface for easy adoption

### 🔍 SLEDgeHammer Evaluation Suite
- **Semantic Pattern Discovery**: Identifies significant cluster-defining features
- **Four-Dimensional Assessment**:
  - **Support (S)**: Feature importance strength
  - **Length (L)**: Optimal descriptor complexity
  - **Exclusivity (E)**: Cluster uniqueness
  - **Difference (D)**: Pattern distribution
- **Intelligent Diagnostics**:
  - Automatic plots
  - Cluster quality visualizations
  - Optimal cluster count recommendations

### 🏗️ FORGE Clustering Engine
- **Binary Data Specialist**: Optimized for categorical feature spaces
- **Smart Exemplar Selection**: Automatically identifies ideal cluster representatives
- **Robust Architecture**: Handles duplicates and edge cases gracefully

---

## 📦 Installation

Get started with pip:

```bash
pip install git+https://github.com/aquinordg/blacksmithing.git
```

---

## 🧪 Usage


```python
# 1. Import
from blacksmithing import FORGE, Sledgehammer
import numpy as np

# 2. Prepare data
X = np.random.randint(0, 2, (100, 5))

# 3. Initialize
model = FORGE(n_clusters=4)

# 4. Cluster
labels = model.fit_predict(X)

# 5. Fit and "predict" new data
model = FORGE(n_clusters=4)
model.fit(X)
new_samples = np.random.randint(0, 2, (5, 5))
new_labels = model.predict(new_samples)

# 6. Evaluate
evaluator = Sledgehammer()
print(f"Quality Score: {evaluator.score(X, labels):.2f}")

# 7. Interpret
for cluster_id, features in evaluator.semantic_descriptors(X, labels, report_form=True).items():
    print(f"\nCluster {cluster_id} Top Features:")
    print(features.head(3))

# 8. Determine best number of clusters (k=2 to k=10)
optimal_k_results = evaluator.find_optimal_k(data=X, clusterer=FORGE(random_state=42), k_max=10, plot=True)
best_k = max(optimal_k_results, key=optimal_k_results.get)
print(f"Optimal number of clusters: {best_k}")
```

---

## 🧱 Architecture Overview

### 🛠️ Core Classes

#### `FORGE(n_clusters=2, evaluator=None, random_state=None)`
*The clustering engine*  
**Key Methods**:
- `fit(X)` - Trains the model on binary/categorical data
- `predict(X)` - Assigns new data points to clusters
- `fit_predict(X)` - Combined training and prediction
- `score(X, labels)` - Returns model's self-evaluation score

#### `Sledgehammer(W=[0.3,0.1,0.5,0.1], particular_threshold=None, aggregation='median')`  
*The evaluation toolkit*  
**Key Methods**:
- `score(X, labels)` - Returns overall clustering quality score (0-1)
- `score_clusters(X, labels)` - Returns detailed scores per cluster
- `semantic_descriptors(X, labels)` - Shows defining features for each cluster
- `find_optimal_k(data, clusterer, k_max)` - Recommends best number of clusters

### 📊 Key Functions

#### Evaluation Metrics
| Function | Description | Returns |
|----------|-------------|---------|
| `score()` | Overall quality score | Float (0-1) |
| `score_clusters()` | Detailed SLED metrics per cluster | DataFrame |

#### Cluster Analysis
| Function | Description | Returns |
|----------|-------------|---------|
| `semantic_descriptors()` | Top features per cluster | Dict/DataFrame |
| `find_optimal_k()` | Optimal cluster count detection | Dict {k: score} |

#### Clustering Operations
| Function | Description | Returns |
|----------|-------------|---------|
| `fit()` | Model training | None |
| `predict()` | Cluster assignments | Array |
| `fit_predict()` | Combined training+prediction | Array |

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing

We welcome contributions to Blacksmithing! To contribute:
1. Fork this repository.
2. Create a new branch for your feature.
3. Submit a pull request with your changes.

For questions or information, feel free to reach out at: [aquinordga@gmail.com](mailto:aquinordga@gmail.com).

---

## 👨‍🔬 Author

Developed by AQUINO, R. D. G. 
[![Lattes](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-plataforma-lattes-32.png)](http://lattes.cnpq.br/2373005809061037)
[![ORCID](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-orcid-32.png)](https://orcid.org/0000-0002-8486-8354)
[![Google Scholar](https://github.com/aquinordg/custom_tools/blob/main/icons/icons8-google-scholar-32.png)](https://scholar.google.com/citations?user=r5WsvKgAAAAJ&hl)

---

## 💡 Feedback

Feel free to open an issue or contact me for feedback or feature requests. Your input is highly appreciated!


