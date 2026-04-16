# K-Means Clustering with External Validation (C++)

## Overview

This project implements the K-Means clustering algorithm from scratch in C++ and evaluates clustering performance using external validation metrics. The program reads labeled datasets, applies clustering, and compares predicted clusters with true labels using multiple evaluation indices.

## Features

* Custom implementation of K-Means clustering (no STL-based algorithms)
* Min-Max normalization for feature scaling
* Random partition initialization
* Multiple clustering runs for improved results
* External validation using:

  * Rand Index
  * Jaccard Index
  * Fowlkes-Mallows Index
* Efficient Euclidean distance computation (without square root)

## Technologies Used

* C++
* File I/O
* Standard libraries (vector, algorithm, etc.)

## How It Works

### 1. Data Input

* Reads dataset from a file
* Format:

  * Number of data points
  * Number of dimensions
  * Number of true clusters
  * Feature values + true label per point

### 2. Preprocessing

* Applies **Min-Max normalization** to scale features between 0 and 1

### 3. Clustering

* Uses **K-Means algorithm** with:

  * Random partition initialization
  * Iterative centroid updates
  * Convergence based on SSE threshold

### 4. Evaluation

After clustering, results are compared to true labels using:

* **Rand Index** – measures overall clustering accuracy
* **Jaccard Index** – measures similarity between clusters
* **Fowlkes-Mallows Index** – balances precision and recall

The algorithm runs multiple times and reports the **best scores**.

## How to Run

### 1. Compile

```bash id="kmc01"
g++ main.cpp -o kmeans
```

### 2. Run

```bash id="kmc02"
./kmeans <filename> <threshold> <runs>
```

### Example

```bash id="kmc03"
./kmeans dataset.txt 0.001 100
```

## Parameters

* `filename` – Input dataset file
* `threshold` – Convergence threshold for SSE
* `runs` – Number of clustering iterations

## Output

The program prints the best evaluation scores across all runs:

```id="kmout01"
Best Rand Index: 0.85
Best Jaccard Index: 0.72
Best Fowlkes-Mallows Index: 0.78
```

## Key Components

* `min_max()` – Normalizes dataset
* `random_partition()` – Initializes centroids
* `kMeans()` – Core clustering algorithm
* `cal_SSE()` – Computes clustering error
* `count_pairs()` – Computes TP, TN, FP, FN
* Validation metrics functions

## Challenges

* Implementing K-Means without high-level libraries
* Efficient pairwise comparison for validation metrics
* Handling convergence and stability across multiple runs

## Future Improvements

* Add alternative initialization methods (e.g., Maximin)
* Implement internal validation indices (Silhouette, CH, Dunn)
* Optimize performance for large datasets
* Add visualization of clusters
* Support additional distance metrics

## Author

Malcolm Howard
University of Central Arkansas
