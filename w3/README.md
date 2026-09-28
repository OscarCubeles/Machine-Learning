# Work3-IML

## Team
- Elena Blanco Lopez
- Clàudia Boixader Garcia
- Óscar Cubeles Ollé
- Alba Fernández Coronado

## Set Up Environment
These sections show how to create a virtual environment for 
our script and how to install dependencies.

```bash
# 1. Open folder in terminal 
cd <root_folder_of_project>/ 

# 2. Create virtual env 
py -m venv venv/ 

# 3. Open virtual env 
venv\Scripts\activate

# 4. Install required dependencies 
pip install -r requirements.txt 

# 5. You can check if dependencies were installed by running the next 
# command, which should print a list of installed dependencies:
pip list 
```

## Run the Code
```bash
# 1. With the environment activated, run the main script
py main.py 
```
## Execute the Script

When you run the main script, the first thing that appears is a menu that asks the user to choose between the two datasets available for analysis: `hepatitis`, `mx` or `cmc`. This choice determines the dataset that the program will process.

After selecting the dataset, the main menu will appear, prompting the user to input the corresponding number for the mode they wish to execute. The options available are as follows:

```plaintext
---------------------------------------------------------
        Work 3 - Introduction to Machine Learning
---------------------------------------------------------

Type the name dataset to be analysed ('hepatitis', 'mx' or 'cmc'): [user input]
```

After selecting the dataset, the main menu will appear, prompting the user to input the corresponding number for the mode they wish to execute. The input must be an integer from `1 to 6`. The options available are as follows:

```plaintext
Type the number of the mode to be executed:

	1) Optics Clustering
	2) Spectral Clustering
	3) K-Means
	4) Improved K-Means
	5) Fuzzy Clustering
	6) Exit 

Choose a mode: 
```

### Mode 1: Optics Clustering

#### Functionality:
In this mode, OPTICS clustering will be executed with different distance metrics and different algorithms to see the 
clustering result as number of clusters, number of noise points and cluster density. 
Once the results for all combinations have been executed, the user will be able to select a combination to see its results. 
These are the possible values: 
  - Distance metric: `[euclidean, cosine, l1]`
  - Algorithms: `['ball_tree', 'brute']` 

#### Example Output

```plaintext
Choose a mode: 1
╒═══════════╤═════════════╤════════════════════╤════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╤═════════════╤══════════════╕
│ metric    │ algorithm   │   silhouette_score │   connectivity │ cluster_density                                                                                                                                                                                    │   #clusters │   #noise_pts │
╞═══════════╪═════════════╪════════════════════╪════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╪═════════════╪══════════════╡
│ euclidean │ ball_tree   │           -1       │       0.993805 │ {0: '1.116e-01'}                                                                                                                                                                                   │           1 │            0 │
├───────────┼─────────────┼────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┼──────────────┤
│ euclidean │ brute       │           -1       │       0.993805 │ {0: '1.116e-01'}                                                                                                                                                                                   │           1 │            0 │
├───────────┼─────────────┼────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┼──────────────┤
│ cosine    │ brute       │            0.33726 │       0.06456  │ {0: '5.581e-02', 1: '6.292e-02', 2: '5.300e-02', 3: '5.119e-02', 4: '7.254e-02', 5: '7.254e-02', 6: '7.254e-02', 7: '6.009e-02', 8: '6.009e-02', 9: '6.009e-02', 10: '6.009e-02', 11: '6.009e-02'} │          12 │          516 │
├───────────┼─────────────┼────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┼──────────────┤
│ l1        │ ball_tree   │           -1       │       0.734892 │ {0: '1.116e-01'}                                                                                                                                                                                   │           1 │            0 │
├───────────┼─────────────┼────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────┼──────────────┤
│ l1        │ brute       │           -1       │       0.734892 │ {0: '1.116e-01'}                                                                                                                                                                                   │           1 │            0 │
╘═══════════╧═════════════╧════════════════════╧════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╧═════════════╧══════════════╛
```

```plaintext
Please enter your command in the following format:
<distance_metric>-<algorithm> (e.g., euclidean-ball_tree)
Enter your command: cosine-brute
```

### Mode 2: Spectral Clustering

#### Functionality:
XXXX

#### Example Output

```plaintext
XXXX
```

### Mode 3: K-Means

#### Functionality:
In this mode, an internal and external analysis will be executed for the K-Means algorithm. 
This mode tests multiple distance metrics (`euclidean`, `manhattan`, `cosine`) and cluster counts (`k_values`). 

For each combination, it runs the algorithm multiple times (`n_runs`), computes average internal evaluation metrics (Silhouette score, Davies-Bouldin score, Calinski-Harabasz score), and stores the results. 
Once computed the metrics, the function visualizes the results to identify the best clustering parameters.

Then, it runs the external analysis, evaluating the performance of K-Means clustering with known labels `y`. 
It selects predefined cluster counts (`k1`, `k2`, `k3`) based on the dataset and runs the algorithm multiple times (`n_runs`). 
After running the algorithms multiple times, it computes external clustering evaluation metrics such as Purity, F1 Score, and Jaccard Index and these are summarized in a table.
Finally, once both internal and external metrics have been evaluated, the user can input a configuration of a `k` value and a distance metric to run the clustering and visualize it using PCA.

The possible distance metrics are: `euclidean`, `manhattan`, and `cosine`. The values for `k` can range between `1` and `10`.
#### Example Output

```plaintext
Choose a mode: 3
100%|██████████| 3/3 [00:16<00:00,  5.64s/it]
╒═══════════╤════════════╤══════════╤════════════╤═════════════════╕
│           │ Distance   │   Purity │   F1 score │   Jaccard Index │
╞═══════════╪════════════╪══════════╪════════════╪═════════════════╡
│ Model k=2 │ Cosine     │ 0.855508 │   0.855508 │        0.619213 │
├───────────┼────────────┼──────────┼────────────┼─────────────────┤
│ Model k=3 │ Cosine     │ 0.841202 │   0.158798 │        0.424925 │
├───────────┼────────────┼──────────┼────────────┼─────────────────┤
│ Model k=5 │ Cosine     │ 0.795422 │   0.427754 │        0.362414 │
╘═══════════╧════════════╧══════════╧════════════╧═════════════════╛
```
```plaintext
Select the configuration to see the final clustering with PCA.
Please enter your command in the following format:
<k>-<distance_metric> (e.g., 2-euclidean)
Enter your command: 2-euclidean
```

### Mode 4: Improved K-Means

#### Functionality:
This mode allows the user to run the two improved K-Means algorithm. The K-Means++ and the X-Means.
First, this mode will ask the user to enter a number in the following range `[1, 2, 3, 4]`.

In sub-modes `1` and `2`, the user will be able to execute the K-Means++ and the X-Means algorithms respectively and their internal and external analysis, as in the K-Means algorithm.

Sub-mode `3` allows the user to run a comparison between all the K-Means algorithms. No other input will be required, the comparison plots will appear.

Finally, `4` can be used to go back to the main menu.

In sub-modes `1` and `2`, he only difference with respect to the K-Means is that for the X-Means, after running both analysis, 
the user will be asked to enter the minimum and maximum k values to execute the clustering.

Next there is an example output.

#### Example Output

```plaintext
Choose a mode: 4

Choose an Improved K-Means version:

	1) KMeans++
	2) X-KMeans
	3) Comparison
	4) Return to Main Menu

Your choice: 2
100%|██████████| 3/3 [00:14<00:00,  4.68s/it]
╒═══════════════╤════════════╤══════════╤════════════╤═════════════════╕
│               │ Distance   │   Purity │   F1 score │   Jaccard Index │
╞═══════════════╪════════════╪══════════╪════════════╪═════════════════╡
│ Model k=(2,7) │ Cosine     │ 0.854077 │   0.854077 │        0.616753 │
╘═══════════════╧════════════╧══════════╧════════════╧═════════════════╛

Select the configuration to see the final clustering with PCA.
Please enter your command in the following format:
<k_min>-<k_max>-<distance_metric> (e.g., 2-10-euclidean)
Enter your command: 2-10-euclidean
```

### Mode 5: Fuzzy Clustering

#### Functionality:
XXXX

#### Example Output

```plaintext
XXXX
```

### Mode 6: Exit

#### Functionality:
In this mode, the user can properly exit the program. 

```plaintext
Choose a mode: 6
Exiting the program. Goodbye!

