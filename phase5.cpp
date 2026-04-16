// coding style: K&R (one true brace style)
// https://github.com/PoshCode/PowerShellPracticeAndStyle/issues/81

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <algorithm>

using namespace std;

// parse for the command line
void args(int argc, char *argv[], string &filename, double &thres, int &runs) {
    // error code if not the right number of arguments
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <filename> <threshold> <runs>\n";
        exit(EXIT_FAILURE);
    }

    // listing the variables
    filename = argv[1];
    thres = stod(argv[2]); // stod to change to a double
    runs = stoi(argv[3]);
}

// read dataset (true clusters added)
vector<int> read_dataset(const string &filename, vector<vector<double> > &dataset, int &num_points, int &dim_points, int &true_clusters) {
    ifstream file(filename);
    vector<int> true_labels;

    if (!file) {
        cerr << "Error opening file!\n";
        exit(EXIT_FAILURE);
    }

    // reads first line
    file >> num_points >> dim_points >> true_clusters;
    dim_points--;

    // reads each point
    for (int i = 0; i < num_points; i++) {
        vector<double> p(dim_points);
        for (int j = 0; j < dim_points; j++) {
            file >> p[j];
        }
        int label;
        file >> label;
        dataset.push_back(p);
        true_labels.push_back(label);
    }

    return true_labels;
}

// normalization using min-max
void min_max(vector<vector<double> > &dataset) {
    int dim_points = dataset[0].size(); // num of features

    // iterate through each column to initialize min and max values
    for (int j = 0; j < dim_points; j++) {
        double minVal = numeric_limits<double>::max();
        double maxVal = numeric_limits<double>::lowest();
        
        // find values for the current column
        for (size_t i = 0; i < dataset.size(); i++) {
            minVal = min(minVal, dataset[i][j]);
            maxVal = max(maxVal, dataset[i][j]);
        }
        
        // perform min-max
        for (size_t i = 0; i < dataset.size(); i++) {
            if (maxVal - minVal != 0)
                dataset[i][j] = (dataset[i][j] - minVal) / (maxVal - minVal);
        }
    }
}

// random partitioning
vector<vector<double> > random_partition(const vector<vector<double> > &dataset, int k) {
    int num_points = dataset.size(); // num of data points
    int dim_points = dataset[0].size(); // num of columns
    
    // create vector to store the cluster centroids, sizes, and assignments
    vector<int> clusterAssignment(num_points);
    vector<vector<double> > centroids(k, vector<double>(dim_points, 0.0)); //start them at 0
    vector<int> clusterSizes(k, 0);
    
    srand(time(0));

    // randomly assign each data point to one of the k clusters
    for (int i = 0; i < num_points; i++) {
        int cluster = rand() % k; // selects random cluster
        clusterAssignment[i] = cluster;
        clusterSizes[cluster]++;


        for (int j = 0; j < dim_points; j++) {
            centroids[cluster][j] += dataset[i][j];
        }
    }
    
    // calculate final centroid by average
    for (int i = 0; i < k; i++) {
        if (clusterSizes[i] > 0) {
            for (int j = 0; j < dim_points; j++) {
                centroids[i][j] /= clusterSizes[i];
            }
        }
    }
    return centroids;
}

// euclidean distance without square root
double eucli_distance(const vector<double> &p1, const vector<double> &p2) {
    double sum = 0.0;
    for (size_t i = 0; i < p1.size(); i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

// assign each point to the nearest centroid
void assign_clusters(const vector<vector<double> > &dataset, vector<int> &cluster_assignments, const vector<vector<double> > &centroids) {
    // go through each point of the dataset
    for (size_t i = 0; i < dataset.size(); i++) {
        double min_Dist = numeric_limits<double>::max(); // start it with large value
        int best_Cluster = 0;
        for (size_t j = 0; j < centroids.size(); j++) { // compare distances for all centroids
            double dist = eucli_distance(dataset[i], centroids[j]);
            if (dist < min_Dist) { //update to shortest distance
                min_Dist = dist;
                best_Cluster = j; 
            }
        }
        cluster_assignments[i] = best_Cluster; // assign point to nearest cluster
    }
}

// Compute new centroids
vector<vector<double> > com_new_centroids(const vector<vector<double> > &dataset, const vector<int> &cluster_assignments, int clusters, int dim_points) {
    vector<vector<double> > new_Centroids(clusters, vector<double>(dim_points, 0.0));
    vector<int> counts(clusters, 0);

    // sum up all coordinates for each cluster
    for (int i = 0; i < dataset.size(); i++) {
        int cluster = cluster_assignments[i];
        for (int j = 0; j < dim_points; j++) {
            new_Centroids[cluster][j] += dataset[i][j]; // Sum coordinates
        }
        counts[cluster]++;
    }

    // divide sum
    for (int i = 0; i < clusters; i++) {
        if (counts[i] > 0) {  // only compute mean if cluster has points
            for (int j = 0; j < dim_points; j++) {
                new_Centroids[i][j] /= counts[i];  // average position
            }
        }
    }
    return new_Centroids;
}

// Sum of Squared Errors
double cal_SSE(const vector<vector<double> > &dataset, const vector<int> &cluster_assignments, const vector<vector<double> > &centroids) {
    double sse = 0.0;
    for (size_t i = 0; i < dataset.size(); ++i) {
        sse += eucli_distance(dataset[i], centroids[cluster_assignments[i]]);
    }
    return sse;
}

// K-Means clustering algorithm implementation
vector<vector<double> > kMeans(vector<vector<double> > dataset, int clusters, int maxIterations, double threshold) {
    int num_points = dataset.size();
    int dim_points = dataset[0].size();
    
    vector<vector<double> > centroids = random_partition(dataset, clusters);
    vector<int> cluster_assignments(dataset.size(), -1);
    double prev_SSE = numeric_limits<double>::max();


    // repeat this until convergence or iterations reached
    for (int iter = 0; iter < maxIterations; iter++) {
        assign_clusters(dataset, cluster_assignments, centroids); // assign cluster to closest centroid
        
        centroids = com_new_centroids(dataset, cluster_assignments, clusters, dim_points); // update centroid for clusters 
        double current_SSE = cal_SSE(dataset, cluster_assignments, centroids); // calculate current SSE

        if (abs(prev_SSE - current_SSE) < threshold) 
            break; // stop if SSE change is small
        prev_SSE = current_SSE;
    }
    return centroids;
}

// counts relationships between predicted and true clusters
void count_pairs(const vector<int> &true_labels, const vector<int> &pred_labels, int &a, int &b, int &c, int &d) {
    int n = true_labels.size();
    a = b = c = d = 0;

    // compare all pairs
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            bool same_true = (true_labels[i] == true_labels[j]);
            bool same_pred = (pred_labels[i] == pred_labels[j]);
            if (same_true && same_pred) a++;            // true positive
            else if (!same_true && !same_pred) b++;     // true negative
            else if (!same_true && same_pred) c++;      // false positive
            else if (same_true && !same_pred) d++;      // false negative
        }
    }
}

double rand_index(int a, int b, int c, int d) {
    return (double)(a + b) / (a + b + c + d);
}

double jaccard_index(int a, int c, int d) {
    return (double)a / (a + c + d);
}

double fowlkes_mallows_index(int a, int c, int d) {
    return (double)a / sqrt((a + c) * (a + d));
}

int main(int argc, char *argv[]) {
    srand(time(0));

    string filename;
    int runs;
    double thres;

    // parse command-line arguments
    args(argc, argv, filename, thres, runs);

    int num_points, dim_points, true_clusters;
    vector<vector<double> > dataset;
    vector<int> true_labels = read_dataset(filename, dataset, num_points, dim_points, true_clusters);

    // normalization method
    min_max(dataset);

    // variables to store best values for each index
    double best_rand = -1.0, best_jaccard = -1.0, best_fmi = -1.0;

    // perform clustering R times
    for (int r = 0; r < runs; r++) {
        vector<int> cluster_assignments(dataset.size(), -1);
        vector<vector<double> > centroids = random_partition(dataset, true_clusters);

        // run k-means clustering
        centroids = kMeans(dataset, true_clusters, runs, thres);

        // reassign clusters after final centroids from kMeans
        assign_clusters(dataset, cluster_assignments, centroids);

        // count TP, TN, FP, FN between predicted and true clusters
        int a, b, c, d;
        count_pairs(true_labels, cluster_assignments, a, b, c, d);

        // compute external validation indices
        double rand = rand_index(a, b, c, d);
        double jac = jaccard_index(a, c, d);
        double fmi = fowlkes_mallows_index(a, c, d);

        if (rand > best_rand) best_rand = rand;
        if (jac > best_jaccard) best_jaccard = jac;
        if (fmi > best_fmi) best_fmi = fmi;
    }

    // output results
    cout << "Best Rand Index: " << best_rand << "\n";
    cout << "Best Jaccard Index: " << best_jaccard << "\n";
    cout << "Best Fowlkes-Mallows Index: " << best_fmi << "\n";

    return 0;
}
