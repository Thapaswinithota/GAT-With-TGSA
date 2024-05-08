import json
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Parse the Log File
def parse_metrics(log_file_path):
    with open(log_file_path, 'r') as file:
        metrics_data = json.load(file)
    return metrics_data

# Step 2: Extract Metrics
def extract_metrics(metrics_data):
    train_metrics = metrics_data['metric']['train']
    valid_metrics = metrics_data['metric']['valid']
    test_metrics = metrics_data['metric']['test']
    return train_metrics, valid_metrics, test_metrics

# Step 3: Visualize Metrics
def visualize_metrics(train_metrics, valid_metrics, test_metrics):
    metrics = ['RMSE', 'MAE', 'pearson', 'R2']
    train_values = [train_metrics[metric] for metric in metrics]
    valid_values = [valid_metrics[metric] for metric in metrics]
    test_values = [test_metrics[metric] for metric in metrics]

    x = np.arange(len(metrics))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, train_values, width, label='Train')
    rects2 = ax.bar(x, valid_values, width, label='Valid')
    rects3 = ax.bar(x + width, test_values, width, label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Metrics by dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.show()

# Main execution
if __name__ == '__main__':
    # Adjust the path to where your log file is located
    log_file_path = 'c:\\Users\\THAPASWINI\\Downloads\\TGSA\\best_metric.log'
    
    # Parse the metrics
    metrics_data = parse_metrics(log_file_path)
    
    # Extract metrics for training, validation, and testing
    train_metrics, valid_metrics, test_metrics = extract_metrics(metrics_data)
    
    # Visualize the metrics
    visualize_metrics(train_metrics, valid_metrics, test_metrics)
