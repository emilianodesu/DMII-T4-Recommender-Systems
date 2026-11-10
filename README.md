# Sistemas de Recomendación a Gran Escala: Reto Netflix & BindingDB (Tema 4)
### Minería de Datos II (Licenciatura en Ciencia de Datos)

Este repositorio contiene las implementaciones de sistemas de recomendación para dos dominios complejos a gran escala, enfocadas al 100% en el **Tema 4 del temario (Sistemas de Recomendación)** y evaluadas con **datos experimentales e históricos reales** (sin datos sintéticos):

1. **Reto Netflix (`netflix_recommender/`)**: Predicción de calificaciones a partir de 100M+ de registros históricos reales del Netflix Prize, evaluada con la partición oficial `probe.txt`.
2. **BindingDB (`bindingdb_recommender/`)**: Predicción de afinidad quimiogenómica fármaco-proteína a partir de bioensayos reales con mediciones experimentales de $IC_{50} \to pIC_{50} = 9 - \log_{10}(IC_{50})$, evaluada en regímenes de pares cálidos (*Random Split*), moléculas nuevas con esqueletos no vistos (*Cold-Scaffold Split* con Bemis-Murcko) y proteínas diana nuevas (*Cold-Target Split*).

Ambos sistemas aprovechan aceleración por hardware en **Apple Silicon (GPU Metal Performance Shaders - `mps`)**, procesamiento columnar out-of-core con **Polars**, paralelización JIT con **Numba**, quimioinformática con **RDKit** y matrices dispersas **SciPy CSR**.

---

## Estructura del Repositorio

```text
├── netflix_recommender/             # Caso de Estudio: Desafío Netflix Prize (Tema 4)
│   ├── src/
│   │   ├── config.py                # Hiperparámetros, rutas y GPU Metal MPS
│   │   ├── data_pipeline.py         # Ingesta streaming de ratings reales y split estricto probe.txt
│   │   ├── evaluation.py            # RMSE oficial, NDCG@K, Recall@K, Cobertura y Novedad
│   │   ├── visualization.py         # Gráficas de convergencia y distribución a 300 DPI
│   │   └── models/
│   │       ├── baseline.py          # 4.1 Baseline predictor regularizado desacoplado
│   │       ├── collaborative_knn.py # 4.3 Item-Item k-NN sobre RESIDUOS con Pearson Shrinkage (Numba CSR)
│   │       ├── latent_factors_gpu.py# 4.1 & 4.4 Biased MF y SVD++ (feedback implícito) en GPU MPS
│   │       ├── content_cold_start.py# 4.2 TF-IDF + Ridge Latent Projection para películas en frío
│   │       └── ensemble.py          # 4.6 Meta-ensamble híbrido stacking (SLSQP)
│   ├── data/                        # Datos reales crudos y particiones procesadas Parquet
│   ├── outputs/                     # 6 figuras de alta resolución y metrics_summary.json
│   ├── tests/                       # Suite de pruebas unitarias
│   ├── netflix_recommender.ipynb    # Notebook ejecutivo pedagógico
│   ├── run_pipeline.py              # CLI maestro (--sample / --full)
│   └── README.md                    # Documentación técnica de resultados reales
│
├── bindingdb_recommender/           # Caso de Estudio: Quimiogenómica BindingDB IC50 (Tema 4)
│   ├── src/
│   │   ├── config.py                # Umbrales bioquímicos (pIC50 >= 6.0) y GPU Metal MPS
│   │   ├── data_pipeline.py         # Curación de IC50 -> pIC50 y splits (Random, Scaffold y Target)
│   │   ├── featurizer.py            # RDKit: Morgan ECFP4, MACCS, Descriptores Lipinski y 3-mers
│   │   ├── evaluation.py            # RMSE, MAE, R², Spearman rho y NDCG@K en los 3 splits
│   │   ├── visualization.py         # Curvas de afinidad, correlación y cribado virtual a 300 DPI
│   │   └── models/
│   │       ├── baseline.py          # 4.1 Baseline bioquímico regularizado
│   │       ├── collaborative_knn.py # 4.3 Filtrado colaborativo sobre residuos centrados (CSR)
│   │       ├── content_chemogenomic.py # 4.2 Similitud Tanimoto ECFP4 x Coseno Secuencia
│   │       ├── latent_factors_gpu.py# 4.4 Biased MF y Two-Tower Hybrid Neural MF en GPU MPS
│   │       └── ensemble.py          # 4.5 Gated Hybrid Ensemble adaptativo (Warm vs Cold)
│   ├── data/                        # TSV real de BindingDB (311 MB) y splits Parquet
│   ├── outputs/                     # 4 figuras diagnósticas y metrics_summary.json
│   ├── tests/                       # Suite de pruebas unitarias de no-leakage y GPU
│   ├── bindingdb_recommender.ipynb  # Notebook interactivo de descubrimiento molecular
│   ├── run_pipeline.py              # CLI maestro (--sample / --full)
│   └── README.md                    # Documentación técnica y validación bioquímica
│
└── requirements.txt                 # Dependencias optimizadas del entorno
```

---

## Resultados y Benchmarks Reales

### 1. Reto Netflix (Evaluación Oficial en `probe.parquet`)

| Modelo | Probe RMSE $\downarrow$ | Probe MAE $\downarrow$ | NDCG@10 $\uparrow$ | Recall@10 $\uparrow$ | Tiempo Entr. (Apple M4) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| *Netflix Global Baseline (Ref)* | 1.0540 | - | - | - | - |
| *Netflix Cinematch Original (Ref)* | 0.9525 | - | - | - | - |
| **Decoupled Baseline** | **0.9158** | 0.7170 | 0.9595 | 1.0000 | 0.05 s |
| **Item-Item k-NN sobre Residuos** | **0.8831** | 0.6847 | 0.9648 | 1.0000 | 0.95 s |
| **Biased Matrix Factorization (MPS)** | **0.8729** | 0.6731 | 0.9641 | 1.0000 | 1.10 s |
| **SVD++ con Feedback Implícito (MPS)** | **0.8618** | 0.6660 | 0.9667 | 1.0000 | 4.30 s |
| **Ensamble Híbrido Stacking** | **0.8556** | **0.6624** | **0.9675** | **1.0000** | **0.02 s** |
| *Netflix 10% Grand Prize Target* | 0.8567 | - | - | - | - |

- **Mejora Monótona:** Cada nivel de modelado reduce el error cuadrático sobre `probe.parquet` sin excepciones.
- **Cold-Start Bridge:** Para películas nuevas sin historial, la proyección Ridge de contenido TF-IDF reduce el RMSE de **1.1298** a **1.0525** (+0.0773 de mejora frente a la media).

---

### 2. BindingDB Quimiogenómico (Predicción de Afinidad $pIC_{50}$)

| Split | Modelo | RMSE $\downarrow$ | MAE $\downarrow$ | $R^2$ $\uparrow$ | Spearman $\rho$ $\uparrow$ | NDCG@10 $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`random_split`** *(Warm-Start)* | Regularized Baseline | 1.263 | 1.025 | 0.249 | 0.500 | 0.626 |
| | Collaborative Residual k-NN | 1.261 | 1.023 | 0.252 | 0.502 | 0.627 |
| | Content Chemogenomic k-NN | **0.878** | **0.687** | **0.637** | **0.810** | **0.923** |
| | **Two-Tower Hybrid Neural MF (MPS)** | **0.787** | **0.601** | **0.709** | **0.843** | **0.918** |
| | **Gated Hybrid Ensemble** | **0.894** | **0.711** | **0.624** | **0.811** | **0.928** |
| **`scaffold_split`** *(Cold-Ligand)* | Regularized Baseline | 1.352 | 1.111 | 0.207 | 0.486 | 0.525 |
| | Content Chemogenomic k-NN | 1.067 | 0.841 | 0.507 | 0.744 | 0.880 |
| | **Two-Tower Hybrid Neural MF (MPS)** | **1.017** | **0.805** | **0.551** | **0.741** | **0.854** |
| | **Gated Hybrid Ensemble** | **1.027** | **0.821** | **0.543** | **0.758** | **0.878** |

- **Meta RMSE < 1.0 Superada:** En `random_split`, el modelo Two-Tower alcanza **RMSE = 0.787** ($R^2 = 0.709$, Spearman $\rho = 0.843$).
- **Scaffold Hopping Exitoso:** En `scaffold_split` (moléculas con esqueleto molecular de Bemis-Murcko nunca visto en entrenamiento), el Two-Tower y Gated Ensemble incrementan la varianza explicada $R^2$ de **0.207 a 0.551** (+166% relativo) y el NDCG@10 de **0.525 a 0.878**.

---

## Instrucciones de Reproducción

```bash
# 1. Activar entorno virtual
source .venv/bin/activate

# 2. Ejecutar pipeline del Reto Netflix
python netflix_recommender/run_pipeline.py --mode sample

# 3. Ejecutar pipeline de BindingDB
python bindingdb_recommender/run_pipeline.py --sample

# 4. Ejecutar pruebas unitarias
python -m unittest discover -s netflix_recommender/tests
python -m unittest discover -s bindingdb_recommender/tests
```
