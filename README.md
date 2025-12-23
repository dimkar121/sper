# SPER: Accelerating Progressive Entity Resolution via Stochastic Bipartite Maximization
This repository contains the source code and datasets for the paper: "SPER: Accelerating Progressive Entity Resolution via Stochastic Bipartite Maximization" by D. Karapiperis (International Hellenic University), G. Papadakis (National and Kapodistrian University of Athens), T. Palpanas (Université Paris Cité; IUF), and V.S. Verykios (Hellenic Open University).

## 📖 Abstract
Entity Resolution is a critical data cleaning task for identifying records that refer to the same real-world entity. In the era of Big Data, traditional batch ER is often infeasible due to volume and velocity constraints, necessitating Progressive ER methods that maximize recall within a limited computational budget. However, existing progressive approaches fail to scale to high-velocity streams because they rely on deterministic sorting to prioritize candidate pairs, a process that incurs prohibitive super-linear complexity and heavy initialization costs. To address this scalability wall, we introduce SPER (Stochastic Progressive ER), a novel framework that redefines prioritization as a sampling problem rather than a ranking problem. By replacing global sorting with a continuous stochastic bipartite maximization strategy, SPER acts as a probabilistic high-pass filter that selects high-utility pairs in strictly linear time. Extensive experiments on eight real-world datasets demonstrate that SPER achieves significant speedups (3x to 6x) over state-of-the-art baselines while maintaining comparable recall and precision.

## 📊 Datasets

The experiments were conducted on a diverse suite of nine real-world and semi-synthetic datasets:

* **Product matching:** ABT-BUY, AMAZON-WALMART, AMAZON-GOOGLE
* **Bibliographic matching:** ACM-DBLP, SCHOLAR-DBLP
* **Movies matching:** IMDB-DBPEDIA
* **Large-Scale Semi-Synthetic:** DBLP, NC-VOTERS

All experiments were run using the `MiniLM-L6-v2` model for embeddings generation.

## ⚙️ Setup and Installation

The implementations rely on several key open-source libraries. You can install them using pip:
 ```bash
    pip install -r requirements.txt
 ```

   
## ▶️ Running the Experiments

The repository is structured to allow for easy replication of the results presented in the paper.

**Running a Single Experiment:** You can run the evaluation for a specific dataset using the corresponding script. For example:
```bash
   python abt.py  
```
uses the ABT-BUY paired dataset.
The large scale datasets—DBLP and NC-VOTERS—can be found [here](https://drive.google.com/drive/folders/1IM9Ot8zpx11YcwXe_4ZTVeEx6wFaHiOo?usp=sharing).


