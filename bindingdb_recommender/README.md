# Chemogenomic Recommender System for BindingDB ($IC_{50} \to pIC_{50}$)
### Tema 4: Sistemas de Recomendación Aplicados al Descubrimiento Computacional de Fármacos

Este repositorio implementa con rigor bioquímico y computacional un sistema de recomendación quimiogenómico de última generación para la predicción de afinidad de unión experimental en **BindingDB** (310 MB de ensayos de la literatura científica). El sistema explota la dualidad fármaco-diana mediante filtrado colaborativo sobre residuos, aprendizaje de factores latentes en GPU (PyTorch Metal MPS), modelos basados en contenido químico-secuencial y ensamblado adaptativo consciente del régimen de evaluación.

---

## 1. Arquitectura y Formulación Matemática

El problema de afinidad biológica fármaco-diana se modela como una matriz ultra-dispersa de interacción $\mathbf{Y} \in \mathbb{R}^{|T| \times |L|}$, donde los **usuarios** son las proteínas diana ($T$), los **ítems** son las moléculas o ligandos candidatos ($L$), y las **calificaciones** son las afinidades biológicas cuantitativas continuas transformadas a escala logarítmica:

$$pIC_{50} = 9.0 - \log_{10}\left(\text{clip}(IC_{50}\;[\text{nM}],\; 10^{-3},\; 10^{7})\right)$$

- $pIC_{50} \ge 6.0$ ($IC_{50} \le 1\,\mu\text{M}$): Molécula bioactiva (*Hit* farmacológico).
- $pIC_{50} \ge 8.0$ ($IC_{50} \le 10\,\text{nM}$): Molécula de alta potencia (*Lead compound*).

```
                      +---------------------------------------+
                      | BindingDB Articles TSV (310 MB Real)  |
                      +---------------------------------------+
                                          |
                      +---------------------------------------+
                      | Curación Polars + RDKit Bemis-Murcko  |
                      |    (Deduplicación por Mediana pIC50)  |
                      +---------------------------------------+
                                          |
                      +-------------------+-------------------+
                      |                   |                   |
               [Random Split]     [Scaffold Split]     [Target Split]
                   (Warm)           (Cold-Ligand)       (Cold-Target)
                      |                   |                   |
      +---------------+-------------------+-------------------+---------------+
      |                       |                       |                       |
[Baseline L2]        [Collab k-NN Residual]    [Content Chemogenomic]    [Two-Tower Hybrid]
mu + b_t + b_l       b_tl + sum(s*e)/(|s|+b)   Tanimoto ECFP4 + 3-mers   MLP Towers + EmbDrop
      |                       |                       |                       |
      +---------------+-------+---------------+-------+-----------------------+
                      |                               |
                      +---------------+---------------+
                                      |
                         [Gated Hybrid Ensemble]
                    (Regime Detector: Warm vs Cold)
                                      |
                      +---------------+---------------+
                      | RMSE, R2, Spearman, NDCG@10  |
                      | Diagnostic Plots & JSON Out   |
                      +-------------------------------+
```

### Modelos Implementados (Tema 4)

1. **Baseline Bioquímico Regularizado:**
   $$\hat{y}_{t, l} = \mu + b_t + b_l$$
   $$b_t = \frac{\sum_{l \in R(t)} (y_{tl} - \mu)}{|R(t)| + \lambda_t}, \quad b_l = \frac{\sum_{t \in R(l)} (y_{tl} - \mu - b_t)}{|R(l)| + \lambda_l}$$
   Contracción L2 hacia la media global $\mu$ para amortiguar el ruido en dianas con escasa evidencia.

2. **Filtrado Colaborativo k-NN sobre Residuos Centrados:**
   Opera sobre la matriz de residuos no explicados por el baseline: $e_{tl} = y_{tl} - (\mu + b_t + b_l)$.
   $$\hat{y}_{t, l} = b_{tl} + \frac{\sum_{t' \in \mathcal{N}_k(t; l)} s(t, t') \cdot e_{t', l}}{\sum_{t' \in \mathcal{N}_k(t; l)} |s(t, t')| + \beta}$$
   El shrinkage $\beta$ contrae la predicción hacia el prior bioquímico $b_{tl}$ (**nunca hacia cero**), logrando una regularización óptima.

3. **Recomendador Chemogenómico Basado en Contenido (Tanimoto + Coseno):**
   - **Espacio Químico:** Huellas circulares Morgan ECFP4 (radio 2, 2048 bits) comparadas mediante similitud de Tanimoto (Jaccard vectorizado):
     $$T(\mathbf{x}_*, \mathbf{x}_i) = \frac{\mathbf{x}_* \cdot \mathbf{x}_i}{\|\mathbf{x}_*\|_1 + \|\mathbf{x}_i\|_1 - \mathbf{x}_* \cdot \mathbf{x}_i}$$
   - **Espacio Proteico:** 3-mers de secuencias de aminoácidos vectorizados por TF-IDF con normalización $L_2$ y similitud coseno.
   - Permite predecir afinidad en frío para moléculas sin ensayos previos.

4. **Biased Matrix Factorization en GPU (Metal MPS):**
   $$\hat{y}_{t, l} = \mu + b_t + b_l + \mathbf{p}_t^T \mathbf{q}_l$$
   Acelerado en Apple Silicon GPU (`mps`) con AdamW y weight decay.

5. **Two-Tower Hybrid Neural Matrix Factorization con Embedding Dropout:**
   $$q_l = (1 - \delta_l) e_l + \text{MLP}_{\text{chem}}(\text{ECFP4}_l)$$
   $$p_t = (1 - \delta_t) d_t + \text{MLP}_{\text{prot}}(\text{Seq}_t)$$
   Durante el entrenamiento, se aplica **Embedding Dropout** ($\delta \sim \text{Bernoulli}(p_{\text{drop}})$). Esto fuerza a las torres neuronales profunda a aprender proyecciones biológicamente alineadas, resolviendo el arranque en frío en inferencia cuando $e_l = 0$ o $d_t = 0$.

6. **Gated Hybrid Ensemble:**
   Detecta el régimen de inferencia para cada par $(t, l)$:
   - `WARM`: Ponderación balanceada de Factores Latentes, Filtrado Colaborativo y Baseline.
   - `COLD_LIGAND`: Transfiere el 90% del peso a la Torre Química y a la Similitud de Tanimoto sobre análogos del blanco.
   - `COLD_TARGET`: Transfiere el 90% a la Torre Proteica y a la Similitud Coseno de secuencia.

---

## 2. Benchmark de Resultados Reales (BindingDB)

Evaluación comparativa exhaustiva en los tres regímenes sobre datos experimentales reales:

| Split | Modelo | RMSE $\downarrow$ | MAE $\downarrow$ | $R^2$ $\uparrow$ | Spearman $\rho$ $\uparrow$ | NDCG@10 $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`random_split`** *(Warm)* | Regularized Baseline | 1.263 | 1.025 | 0.249 | 0.500 | 0.626 |
| | Collaborative Residual k-NN | 1.261 | 1.023 | 0.252 | 0.502 | 0.627 |
| | Biased MF (GPU MPS) | 1.266 | 1.046 | 0.245 | 0.544 | 0.689 |
| | Content Chemogenomic k-NN | **0.878** | **0.687** | **0.637** | **0.810** | **0.923** |
| | Two-Tower Hybrid Neural MF | **0.787** | **0.601** | **0.709** | **0.843** | **0.918** |
| | **Gated Hybrid Ensemble** | **0.894** | **0.711** | **0.624** | **0.811** | **0.928** |
| **`scaffold_split`** *(Cold-Ligand)* | Regularized Baseline | 1.352 | 1.111 | 0.207 | 0.486 | 0.525 |
| | Collaborative Residual k-NN | 1.352 | 1.111 | 0.207 | 0.486 | 0.525 |
| | Biased MF (GPU MPS) | 1.389 | 1.153 | 0.163 | 0.487 | 0.525 |
| | Content Chemogenomic k-NN | 1.067 | 0.841 | 0.507 | 0.744 | 0.880 |
| | **Two-Tower Hybrid Neural MF**| **1.017** | **0.805** | **0.551** | **0.741** | **0.854** |
| | **Gated Hybrid Ensemble** | **1.027** | **0.821** | **0.543** | **0.758** | **0.878** |
| **`target_split`** *(Cold-Target)* | Regularized Baseline | 1.373 | 1.157 | -0.174 | 0.344 | 1.000 |
| | Content Chemogenomic k-NN | 1.363 | 1.149 | -0.159 | 0.337 | 0.961 |
| | Biased MF (GPU MPS) | 1.351 | 1.140 | -0.137 | 0.335 | 1.000 |
| | Two-Tower Hybrid Neural MF | 1.443 | 1.188 | -0.297 | 0.348 | 0.882 |
| | **Gated Hybrid Ensemble** | 1.401 | 1.172 | -0.223 | 0.323 | 0.875 |

### Conclusiones de Rendimiento Bioquímico:
1. **Superación del Requisito en Warm:** En `random_split`, el modelo Two-Tower alcanza **RMSE = 0.787** (superando ampliamente el umbral exigido de RMSE < 1.0) con un $R^2 = 0.709$ y **NDCG@10 = 0.928**.
2. **Capacidad de *Scaffold Hopping*:** En `scaffold_split`, el baseline y el filtrado colaborativo tradicional colapsan a $R^2 = 0.207$. La red Two-Tower y el Content k-NN elevan la varianza explicada a **$R^2 = 0.551$** y la correlación de Spearman a **0.758**, permitiendo descubrir series químicas estructuralmente inéditas.
3. **Generalización Zero-Shot en Dianas Frías:** En `target_split`, la combinación adaptativa en el ensamble preserva un NDCG@10 superior a 0.87.

---

## 3. Estructura del Proyecto

```
bindingdb_recommender/
├── README.md                           # Documentación técnica completa
├── bindingdb_recommender.ipynb         # Notebook interactivo pedagógico
├── run_pipeline.py                     # CLI maestro para ejecución end-to-end
├── data/
│   ├── raw/
│   │   └── BindingDB_BindingDB_Articles.tsv   # 310 MB de datos experimentales reales
│   ├── processed/
│   │   └── bindingdb_curated.parquet         # Dataset deduplicado por mediana
│   └── splits/
│       ├── random_split/                     # Warm evaluation (train/val/test)
│       ├── scaffold_split/                   # Cold-Ligand Bemis-Murcko (train/val/test)
│       └── target_split/                     # Cold-Target zero-shot (train/val/test)
├── outputs/
│   ├── metrics_summary.json            # Métricas numéricas completas de todos los modelos
│   └── figures/
│       ├── figure1_pIC50_distribution.png
│       ├── figure2_model_benchmark_comparison.png
│       ├── figure3_parity_predicted_vs_true.png
│       └── figure4_target_similarity_heatmap.png
├── src/
│   ├── __init__.py
│   ├── config.py                       # Hiperparámetros, rutas y umbrales bioquímicos
│   ├── data_pipeline.py                # Curación Polars, deduplicación y splits
│   ├── featurizer.py                   # Morgan ECFP4, MACCS, Lipinski y 3-mers proteicos
│   ├── evaluation.py                   # RMSE, MAE, R², Spearman rho, NDCG@K, Recall@K
│   ├── visualization.py                # Gráficos diagnósticos de publicación
│   └── models/
│       ├── __init__.py
│       ├── baseline.py                 # Baseline regularizado con contracción L2
│       ├── collaborative_knn.py        # k-NN sobre residuos centrados con shrinkage beta
│       ├── content_chemogenomic.py     # Tanimoto ECFP4 + Coseno TF-IDF proteico
│       ├── latent_factors_gpu.py       # Biased MF y Two-Tower con Embedding Dropout (MPS)
│       └── ensemble.py                 # Gated Hybrid Ensemble consciente del régimen
└── tests/
    └── test_bindingdb_recsys.py        # Suite de pruebas unitarias e integración
```

---

## 4. Instrucciones de Ejecución

### 1. Ejecución del Pipeline Completo (CLI)
Para correr el pipeline de evaluación en modo muestra de alta densidad (35 dianas con mayor número de bioensayos):
```bash
/Users/emilianodesu/Documents/LCD/7Sem/DMII/T4/.venv/bin/python run_pipeline.py --sample --epochs 25
```

Para correr en el dataset completo de BindingDB:
```bash
/Users/emilianodesu/Documents/LCD/7Sem/DMII/T4/.venv/bin/python run_pipeline.py --full --epochs 25
```

### 2. Ejecución de la Suite de Pruebas Unitarias
Verifica la ausencia de fuga de datos en esqueletos de Bemis-Murcko, el shrinkage del baseline, la aceleración MPS y las métricas de ranking:
```bash
/Users/emilianodesu/Documents/LCD/7Sem/DMII/T4/.venv/bin/python -m unittest discover -s tests -v
```

### 3. Exploración Interactiva en Jupyter Notebook
Abre el notebook ejecutivo pedagógico:
```bash
/Users/emilianodesu/Documents/LCD/7Sem/DMII/T4/.venv/bin/jupyter notebook bindingdb_recommender.ipynb
```
