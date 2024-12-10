# imports 
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# define a class to process the data and train and test the models
class TrainAndValidate:
    def __init__(self, grid, data, partitions=[0.2, 0.5, 0.8], n_trials=3):
        self.grid = grid
        self.data = data
        self.partitions = partitions
        self.n_trials = n_trials
        self.results = {}
        self.hp_performance = {}
        self.iteration_accuracies = {}

    def preprocess_data(self, df, target_column, dropna=True):
        """
        Function to preprocess the data

        Parameters:
        - df (pd.DataFrame): dataset to preprocess.
        - target_column (str): The name of the target column.
        - dropna (bool): Whether to drop rows with missing values.

        Returns:
        - X (pd.DataFrame): Preprocessed features.
        - y (pd.Series): Target column.
        """
        # Drop rows with missing values
        if dropna:
            df = df.dropna()

        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # One-hot encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        return X, y

    # function to scale the data (mean = 0, std = 1)
    def scale_data(self, X_train, X_test):
        """
        Scale numerical features using StandardScaler.

        Parameters:
        - X_train (pd.DataFrame): Training features.
        - X_test (pd.DataFrame): Test features.

        Returns:
        - X_train_scaled (pd.DataFrame): Scaled training features.
        - X_test_scaled (pd.DataFrame): Scaled test features.
        """
        scaler = StandardScaler()

        # Fit on training data and transform both training and test data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        return X_train_scaled, X_test_scaled

    def train_and_validate(self):
        """
        Method to train and validate classifiers on multiple datasets, partitions, and trials.
        """
        for dataset_name, (X, y) in self.data.items():
            print(f"Dataset: {dataset_name}")
            self.results[dataset_name] = {}
            self.hp_performance[dataset_name] = {}
            self.iteration_accuracies[dataset_name] = {}
            X, y = self.preprocess_data(pd.concat([X, y], axis = 1), target_column=y.columns[0])

            # loop over the classifiers and corresponding hyperparameters
            for clf, params in self.grid.items():
                print(f"Classifier: {clf.__name__}")
                self.results[dataset_name][clf.__name__] = {}
                self.hp_performance[dataset_name][clf.__name__] = {}
                self.iteration_accuracies[dataset_name][clf.__name__] = {partition: [] for partition in self.partitions}


                # loop over the partitions of train and test data
                for partition in self.partitions:
                    print(f"Partition: {partition} training, {1 - partition} testing")
                    self.hp_performance[dataset_name][clf.__name__][partition] = {}

                    metrics = {
                        'train_accuracy': [],
                        'validation_accuracy': [],
                        'test_accuracy': [],
                        'test_roc_auc': [],
                        'test_f1': []
                    }
                    hp_performance = {}

                    # for each partition, loop over the trials (default 3)
                    for trial in range(self.n_trials):

                        # split data into training and testing
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=1 - partition, random_state=trial
                        )

                        # scale X_train and X_test
                        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)

                        # Hyperparameter tuning on the training data using 3-fold cross-validation
                        grid_search = GridSearchCV(
                            clf(),
                            {key: value for key, value in params.items()},
                            cv=3,
                            scoring='accuracy',
                            n_jobs=-1,
                            return_train_score=True
                        )
                        grid_search.fit(X_train_scaled, y_train.values.ravel())

                        # save the best model, best parameters, and validation accuracy
                        best_model = grid_search.best_estimator_
                        best_params = grid_search.best_params_
                        validation_acc = grid_search.best_score_
                        for param_set, train_score, val_score in zip(
                            grid_search.cv_results_['params'],
                            grid_search.cv_results_['mean_train_score'],
                            grid_search.cv_results_['mean_test_score']
                        ):
                            param_key = str(param_set)
                            if param_key not in hp_performance:
                                hp_performance[param_key] = {
                                    'train_errors': [],
                                    'validation_errors': []
                                }
                            hp_performance[param_key]['train_errors'].append(1 - train_score)
                            hp_performance[param_key]['validation_errors'].append(1 - val_score)

                        # save trining and testing accuracy, roc_auc, and f1 score
                        train_acc = accuracy_score(y_train, best_model.predict(X_train_scaled))
                        test_acc = accuracy_score(y_test, best_model.predict(X_test_scaled))

                        # Save train/test accuracies for this iteration - use later to plot acc curve
                        self.iteration_accuracies[dataset_name][clf.__name__][partition].append({
                            'trial': trial + 1,
                            'train_accuracy': train_acc,
                            'test_accuracy': test_acc
                        })
                        
                        # Handle ROC-AUC depending on the classifier's capabilities
                        if hasattr(best_model, "predict_proba"):
                            roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
                        elif hasattr(best_model, "decision_function"):
                            roc_auc = roc_auc_score(y_test, best_model.decision_function(X_test))
                        else:
                            roc_auc = None

                        test_f1 = f1_score(y_test, best_model.predict(X_test))

                        metrics['train_accuracy'].append(train_acc)
                        metrics['validation_accuracy'].append(validation_acc)
                        metrics['test_accuracy'].append(test_acc)
                        metrics['test_roc_auc'].append(roc_auc)
                        metrics['test_f1'].append(test_f1)
                    
                    # Average results across trials
                    self.results[dataset_name][clf.__name__][partition] = {
                        'best_params': best_params,
                        'train_accuracy': np.mean(metrics['train_accuracy']),
                        'validation_accuracy': np.mean(metrics['validation_accuracy']),
                        'test_accuracy': np.mean(metrics['test_accuracy']),
                        'test_roc_auc': np.mean(metrics['test_roc_auc']),
                        'test_f1': np.mean(metrics['test_f1'])
                    }
                    self.hp_performance[dataset_name][clf.__name__][partition] = hp_performance

    def plot_performance(self, metric='test_accuracy'):
        """
        Plot the performance of each classifier across all datasets for the selected metric.

        Parameters:
        - metric (str): The metric to plot ('train_accuracy', 'validation_accuracy', 'test_accuracy', 'test_roc_auc', 'test_f1').
        """
        classifiers = set()
        # extract all classifier names from the results
        for dataset_name in self.results:
            classifiers.update(self.results[dataset_name].keys())

        # loop over the classifiers 
        for classifier_name in classifiers:
            plt.figure(figsize=(7, 5))
            datasets = []
            partition_metrics = {partition: [] for partition in self.partitions}

            # extract metrics for each dataset and partition
            for dataset_name, classifiers_data in self.results.items():
                if classifier_name in classifiers_data:
                    datasets.append(dataset_name)
                    for partition in self.partitions:
                        partition_metrics[partition].append(
                            classifiers_data[classifier_name].get(partition, {}).get(metric, None)
                        )

            # Plot each partition's results
            for partition, metrics in partition_metrics.items():
                plt.plot(datasets, metrics, marker='o', linestyle='-', label=f'{int(partition * 100)}% Training')

            # Add plot details
            plt.title(f'{metric.replace("_", " ").capitalize()} for {classifier_name} Across Datasets', fontsize=16)
            plt.xlabel('Dataset', fontsize=14)
            plt.ylabel(metric.replace("_", " ").capitalize(), fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(title='Partition', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 1.2)
            sns.despine()
            plt.tight_layout()
            plt.show()
    def plot_comparison(self, metric='test_accuracy'):
        """
        Plot the performance of classifiers to determine the best one across partitions and datasets.
        
        Parameters:
        - metric (str): The metric to plot ('train_accuracy', 'validation_accuracy', 'test_accuracy', 'test_roc_auc', 'test_f1').
        """
        classifier_partition_scores = {}
        classifier_dataset_scores = {}

        # Collect scores for each classifier
        for _, classifiers_data in self.results.items():
            for classifier_name, partitions_data in classifiers_data.items():
                if classifier_name not in classifier_partition_scores:
                    classifier_partition_scores[classifier_name] = {partition: [] for partition in self.partitions}
                if classifier_name not in classifier_dataset_scores:
                    classifier_dataset_scores[classifier_name] = []

                for partition, metrics in partitions_data.items():
                    if metric in metrics:
                        classifier_partition_scores[classifier_name][partition].append(metrics[metric])
                        classifier_dataset_scores[classifier_name].append(metrics[metric])

        # Compute averages for plotting
        avg_partition_scores = {
            clf: {partition: np.mean(scores) if scores else 0
                for partition, scores in partitions.items()}
            for clf, partitions in classifier_partition_scores.items()
        }
        avg_dataset_scores = {
            clf: np.mean(scores) if scores else 0
            for clf, scores in classifier_dataset_scores.items()
        }

        # Plot classifier performance across partitions
        plt.figure(figsize=(10, 6))
        for clf, scores in avg_partition_scores.items():
            partitions = [f"{int(p * 100)}%" for p in scores.keys()]
            averages = list(scores.values())
            plt.plot(partitions, averages, marker='o', linestyle='-', label=clf)

        plt.title(f'{metric.replace("_", " ").capitalize()} Across Partitions', fontsize=16)
        plt.xlabel('Partition (Training Data Proportion)', fontsize=14)
        plt.ylabel(metric.replace("_", " ").capitalize(), fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Classifier', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        sns.despine()
        plt.tight_layout()
        plt.show()

        # Plot classifier performance across datasets
        plt.figure(figsize=(12, 8))
        classifiers = list(avg_dataset_scores.keys())
        averages = list(avg_dataset_scores.values())
        sns.barplot(x=classifiers, y=averages, palette='viridis')
        # Add metric values on top of bars
        for i, value in enumerate(averages):
            plt.text(i, value + 0.02, f"{value:.2f}", ha='center', fontsize=12, color='black', fontweight='bold')
        plt.title(f'{metric.replace("_", " ").capitalize()} Across Datasets', fontsize=16)
        plt.xlabel('Classifier', fontsize=14)
        plt.ylabel(metric.replace("_", " ").capitalize(), fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        sns.despine()
        plt.tight_layout()
        plt.show()

    def plot_accuracy_curves(self, dataset_name, classifier_name, partition, plot_train=True, plot_test=True):
        """
        Plot accuracy curves for a specific dataset, classifier, and partition.

        Parameters:
        - dataset_name (str): The name of the dataset.
        - classifier_name (str): The name of the classifier (e.g., "RandomForestClassifier").
        - partition (float): The training partition (e.g., 0.2, 0.5, 0.8).
        - plot_train (bool): If True, plot the training accuracy curve.
        - plot_test (bool): If True, plot the test accuracy curve.
        """
        # Retrieve data for the specified dataset, classifier, and partition
        partition_data = self.iteration_accuracies.get(dataset_name, {}).get(classifier_name, {}).get(partition, None)
        
        if not partition_data:
            print(f"No data found for dataset '{dataset_name}', classifier '{classifier_name}', and partition {partition}.")
            return

        # Extract trials, train accuracies, and test accuracies
        trials = [item['trial'] for item in partition_data]
        train_accuracies = [item['train_accuracy'] for item in partition_data]
        test_accuracies = [item['test_accuracy'] for item in partition_data]

        # Plot the selected curves
        plt.figure(figsize=(10, 6))
        if plot_train:
            plt.plot(trials, train_accuracies, marker='o', label='Train Accuracy', linestyle='-')
        if plot_test:
            plt.plot(trials, test_accuracies, marker='o', label='Test Accuracy', linestyle='--')

        # Plot details
        plt.title(f"Accuracy Curves for {classifier_name} on {dataset_name} ({int(partition * 100)}% Train)", fontsize=16)
        plt.xlabel('Trial', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


def average_across_datasets(trainer, metric_list=['test_accuracy', 'test_roc_auc', 'test_f1']):
    """
    Create a table showing each classifier performance averaged across datasets for each metric.

    Parameters:
    - trainer: The TrainAndValidate instance containing the results.
    - metric_list (list): List of metrics to include.

    Returns:
    - pd.DataFrame: Table with classifiers as rows and metrics as columns.
    """
    results = []

    for classifier_name in trainer.results[next(iter(trainer.results))].keys():
        metric_scores = {metric: [] for metric in metric_list}

        for dataset_name, classifiers_data in trainer.results.items():
            for partition in trainer.partitions:
                for metric in metric_list:
                    score = classifiers_data[classifier_name][partition].get(metric, None)
                    if score is not None:
                        metric_scores[metric].append(score)

        # Average scores across datasets
        averaged_scores = {metric: np.mean(metric_scores[metric]) for metric in metric_list}
        averaged_scores['Classifier'] = classifier_name
        results.append(averaged_scores)

    return pd.DataFrame(results)

## functions to calculate normalized results similarly to what was done in the paper
def normalize_and_aggregate_results(trainer, metric_list=['test_accuracy', 'test_roc_auc', 'test_f1']):
    """
    Normalize metrics and compute averages across datasets and metrics.

    Parameters:
    - trainer: The TrainAndValidate instance containing the results.
    - metric_list (list): List of metrics to normalize and aggregate.

    Returns:
    - pd.DataFrame: Table summarizing normalized and averaged results.
    """
    results_data = []

    # Collect raw scores
    raw_scores = {metric: [] for metric in metric_list}
    for dataset_name, classifiers_data in trainer.results.items():
        for classifier_name, partitions_data in classifiers_data.items():
            for partition in trainer.partitions:
                for metric in metric_list:
                    score = partitions_data[partition].get(metric, None)
                    if score is not None:
                        raw_scores[metric].append(score)

    # Normalize scores for each metric (min-max scaling)
    normalized_scores = {}
    for metric, scores in raw_scores.items():
        min_score, max_score = min(scores), max(scores)
        normalized_scores[metric] = [(score - min_score) / (max_score - min_score) for score in scores]

    # Aggregate scores
    for dataset_name, classifiers_data in trainer.results.items():
        for classifier_name, partitions_data in classifiers_data.items():
            normalized_metric_scores = {metric: [] for metric in metric_list}
            for partition in trainer.partitions:
                for metric in metric_list:
                    score = partitions_data[partition].get(metric, None)
                    if score is not None:
                        normalized_metric_scores[metric].append(score)

            # Compute averages
            avg_score_across_metrics = np.mean([np.mean(normalized_metric_scores[metric]) for metric in metric_list])
            results_data.append({
                'Classifier': classifier_name,
                'Dataset': dataset_name,
                'Avg Score (Metrics)': avg_score_across_metrics,
            })

    # Create DataFrame
    results_df = pd.DataFrame(results_data)
    return results_df
import pandas as pd

def build_results_table(trainer, metric_list=['test_accuracy', 'test_roc_auc', 'test_f1']):
    """
    Build a summary table showcasing the best classifier for each dataset, partition, and metric.
    
    Parameters:
    - trainer: The TrainAndValidate instance containing the results.
    - metric_list (list): List of metrics to include in the table (default: accuracy, ROC-AUC, F1-score).
    
    Returns:
    - pd.DataFrame: Summary table with the best classifier for each dataset, partition, and metric.
    """
    results_data = []

    for dataset_name, classifiers_data in trainer.results.items():
        for partition in trainer.partitions:
            for metric in metric_list:
                best_classifier = None
                best_score = -float('inf')

                # Find the best classifier for the given dataset, partition, and metric
                for classifier_name, partitions_data in classifiers_data.items():
                    if partition in partitions_data:
                        score = partitions_data[partition].get(metric, None)
                        if score is not None and score > best_score:
                            best_score = score
                            best_classifier = classifier_name
                
                # Append the results
                results_data.append({
                    'Dataset': dataset_name,
                    'Partition': f"{int(partition * 100)}% Training",
                    'Metric': metric,
                    'Best Classifier': best_classifier,
                    'Best Score': best_score
                })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results_data)
    return results_df


def summarize_partition_comparison_all_datasets(trainer, metric='test_accuracy'):

    """
    Create a summary table showing partition comparison for each classifier across all datasets.

    Parameters:
    - trainer: TrainAndValidate instance containing the results.
    - metric (str): Metric to summarize (e.g., 'test_accuracy').

    Returns:
    - pd.DataFrame: Summary table.
    """
    rows = []
    for dataset_name, classifiers_data in trainer.results.items():
        for classifier_name, partitions_data in classifiers_data.items():
            row = {'Classifier': classifier_name, 'Dataset': dataset_name}
            for partition in trainer.partitions:
                row[f"{int(partition * 100)}% Train"] = partitions_data[partition].get(metric, None)
            rows.append(row)
    return pd.DataFrame(rows)

def plot_hyperparam_heatmap(trainer, dataset_name, classifier_name, partition, param_x, param_y, metric='validation_errors'):
    """
    Plot a heatmap of performance metrics (e.g., accuracy) for combinations of two hyperparameters.

    Parameters:
    - trainer: The TrainAndValidate instance containing hyperparameter performance data.
    - dataset_name (str): The name of the dataset.
    - classifier_name (str): The name of the classifier (e.g., "RandomForestClassifier").
    - partition (float): The training partition (e.g., 0.2, 0.5, 0.8).
    - param_x (str): The name of the hyperparameter to plot on the x-axis.
    - param_y (str): The name of the hyperparameter to plot on the y-axis.
    - metric (str): The performance metric to visualize ('validation_errors' or 'train_errors').
    """
    # Retrieve hyperparameter performance data
    hp_data = trainer.hp_performance.get(dataset_name, {}).get(classifier_name, {}).get(partition, {})
    
    if not hp_data:
        print(f"No hyperparameter data available for {dataset_name}, {classifier_name}, partition {partition}.")
        return

    # Prepare data for the heatmap
    records = []
    for params, errors in hp_data.items():
        param_set = eval(params)  # Convert string representation of dict back to a dictionary
        if param_x in param_set and param_y in param_set:
            mean_metric = np.mean(errors[metric])  # Average across trials
            records.append({
                param_x: param_set[param_x],
                param_y: param_set[param_y],
                metric: mean_metric
            })

    if not records:
        print(f"No data found for hyperparameters '{param_x}' and '{param_y}'.")
        return

    # Convert to a DataFrame
    df = pd.DataFrame(records)

    # Pivot data for heatmap
    heatmap_data = df.pivot_table(index=param_y, columns=param_x, values=metric)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", cbar_kws={'label': metric.capitalize()})
    plt.title(f"Hyperparameter Heatmap for {classifier_name} on {dataset_name} ({int(partition * 100)}% Train)", fontsize=16)
    plt.xlabel(param_x, fontsize=14)
    plt.ylabel(param_y, fontsize=14)
    plt.tight_layout()
    plt.show()