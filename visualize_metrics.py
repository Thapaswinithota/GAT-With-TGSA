# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Step 1: Parse the Log File
# def parse_metrics(log_file_path):
#     with open(log_file_path, 'r') as file:
#         metrics_data = json.load(file)
#     return metrics_data

# # Step 2: Extract Metrics
# def extract_metrics(metrics_data):
#     train_metrics = metrics_data['metric']['train']
#     valid_metrics = metrics_data['metric']['valid']
#     test_metrics = metrics_data['metric']['test']
#     return train_metrics, valid_metrics, test_metrics

# # Step 3: Visualize Metrics
# def visualize_metrics(train_metrics, valid_metrics, test_metrics):
#     metrics = ['RMSE', 'MAE', 'pearson', 'R2']
#     train_values = [train_metrics[metric] for metric in metrics]
#     valid_values = [valid_metrics[metric] for metric in metrics]
#     test_values = [test_metrics[metric] for metric in metrics]

#     x = np.arange(len(metrics))  # the label locations
#     width = 0.25  # the width of the bars

#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width, train_values, width, label='Train')
#     rects2 = ax.bar(x, valid_values, width, label='Valid')
#     rects3 = ax.bar(x + width, test_values, width, label='Test')

#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     ax.set_ylabel('Scores')
#     ax.set_title('Metrics by dataset')
#     ax.set_xticks(x)
#     ax.set_xticklabels(metrics)
#     ax.legend()

#     plt.show()

# # Main execution
# if __name__ == '__main__':
#     # The directory and file names
#     log_directory = 'log_20240329_163658'  # Use the actual directory name if different
#     log_file_name = 'best_metric.log'
    
#     # The base directory is where the logs folder is located
#     base_directory = 'c:\\Users\\THAPASWINI\\Downloads\\TGSA\\logs'
    
#     # Construct the full path to the log file
#     log_file_path = os.path.join(base_directory, log_directory, log_file_name)
    
#     # Print the path for verification
#     print(f'Looking for log file at: {log_file_path}')
    
#     # Ensure the file exists before attempting to open it
#     if not os.path.exists(log_file_path):
#         print(f"The file {log_file_path} does not exist. Please check the file path.")
#     else:
#         # Parse the metrics
#         metrics_data = parse_metrics(log_file_path)
        
#         # Extract metrics for training, validation, and testing
#         train_metrics, valid_metrics, test_metrics = extract_metrics(metrics_data)
        
#         # Visualize the metrics
#         visualize_metrics(train_metrics, valid_metrics, test_metrics)



import json
import matplotlib.pyplot as plt
import numpy as np

# Your JSON data here
json_data = """
{"metric": {"epoch": 36, "train": {"RMSE": 0.5467739701271057, "MAE": 0.42593839419900126, "pearson": 0.9810831290909944, "R2": 0.9619107455608294}, "valid": {"RMSE": 0.8684057593345642, "MAE": 0.6428726687733349, "pearson": 0.9509586802743596, "R2": 0.9039959865863232}, "test": {"RMSE": 0.8696364164352417, "MAE": 0.6417876896700175, "pearson": 0.9517756859145734, "R2": 0.905591919700826}}}
"""

# Parse the JSON data
metrics_data = json.loads(json_data)

# Extract the metrics
train_metrics = metrics_data['metric']['train']
valid_metrics = metrics_data['metric']['valid']
test_metrics = metrics_data['metric']['test']

# Define the metrics and their corresponding values
metrics = ['RMSE', 'MAE', 'pearson', 'R2']
train_values = [train_metrics[metric] for metric in metrics]
valid_values = [valid_metrics[metric] for metric in metrics]
test_values = [test_metrics[metric] for metric in metrics]

# Set up the matplotlib figure and axes
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, train_values, width, label='Train', color='blue')
rects2 = ax.bar(x, valid_values, width, label='Valid', color='orange')
rects3 = ax.bar(x + width, test_values, width, label='Test', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Metrics by dataset')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# Attach labels
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

plt.show()

