import os
import matplotlib.pyplot as plt

# Define the path to the log files
log_dir = "./split_training_logs"
log_files = ["DPSGD_SGD_4-8.log", "SGD_DPSGD_4-8.log"]  # Log file names

def parse_log(file_path):
    """ Parses the log file and extracts epoch, test loss, and method for visualization. """
    epochs, test_loss, methods = [], [], []
    current_method = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"'):
                current_method = line.strip('"')  # Extract method name
                continue

            parts = line.split(',')
            epoch = int(parts[0].strip())
            test_l = float(parts[3].strip())

            epochs.append(epoch)
            test_loss.append(test_l)
            methods.append(current_method)

    return epochs, test_loss, methods

# Parse both log files
epochs1, test_loss1, methods1 = parse_log(os.path.join(log_dir, log_files[0]))  # DPSGD -> SGD
epochs2, test_loss2, methods2 = parse_log(os.path.join(log_dir, log_files[1]))  # SGD -> DPSGD

# Define line colors
colors = {"DPSGD_SGD_4-8.log": "blue", "SGD_DPSGD_4-8.log": "red"}
labels = {"DPSGD_SGD_4-8.log": "DPSGD → SGD", "SGD_DPSGD_4-8.log": "SGD → DPSGD"}

# Identify the switch point
switch_epoch1 = epochs1[4]  # DPSGD -> SGD switch happens after epoch 4
switch_epoch2 = epochs2[4]  # SGD -> DPSGD switch happens after epoch 4

# Plotting
plt.figure(figsize=(10, 6))

# Plot test loss from the first log file (DPSGD → SGD)
plt.plot(epochs1, test_loss1, color=colors["DPSGD_SGD_4-8.log"], linewidth=2, linestyle="dashed", label=labels["DPSGD_SGD_4-8.log"])

# Plot test loss from the second log file (SGD → DPSGD)
plt.plot(epochs2, test_loss2, color=colors["SGD_DPSGD_4-8.log"], linewidth=2, linestyle="dashed", label=labels["SGD_DPSGD_4-8.log"])
plt.axvline(x=switch_epoch2, color="black", linestyle="dotted", label="Switch Point")

# Ensure all epochs are numbered on the x-axis
plt.xticks(range(min(epochs1 + epochs2), max(epochs1 + epochs2) + 1), fontsize=12)
plt.yticks(fontsize=12)

# Labels and legend
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Test Loss", fontsize=16)
plt.title("Test Loss Comparison (Mixed Training)", fontsize=20)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()