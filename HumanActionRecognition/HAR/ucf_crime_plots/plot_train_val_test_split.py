import os
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def count_files_in_folders(directory_path):
    """
    Count the number of files in each subfolder of the given directory.
    
    Args:
        directory_path (str): Path to the directory containing class folders
        
    Returns:
        dict: Dictionary with folder names as keys and file counts as values
    """
    folder_file_counts = {}
    
    # Get all items in the directory
    items = os.listdir(directory_path)
    
    # Filter only directories (folders) and exclude hidden/system folders
    folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item)) 
               and not item.startswith('.') and item not in ['.venv', '.qodo', '__pycache__']]
    
    # Count files in each folder
    for folder in folders:
        folder_path = os.path.join(directory_path, folder)
        try:
            # Count only files (not subdirectories)
            file_count = len([f for f in os.listdir(folder_path) 
                            if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.mp4')])
            folder_file_counts[folder] = file_count
        except PermissionError:
            print(f"Permission denied for folder: {folder}")
            folder_file_counts[folder] = 0
    
    return folder_file_counts

def calculate_split_counts(total_count, train_ratio=0.56, val_ratio=0.14, test_ratio=0.30):
    """
    Calculate train, validation, and test split counts based on ratios.
    
    Args:
        total_count (int): Total number of videos
        train_ratio (float): Training set ratio (default: 0.56)
        val_ratio (float): Validation set ratio (default: 0.14)
        test_ratio (float): Test set ratio (default: 0.30)
        
    Returns:
        tuple: (train_count, val_count, test_count)
    """
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
    test_count = total_count - train_count - val_count  # Ensure all videos are allocated
    
    return train_count, val_count, test_count

def prepare_split_data(folder_counts):
    """
    Prepare data for train-validation-test split visualization.
    
    Args:
        folder_counts (dict): Dictionary with folder names and file counts
        
    Returns:
        tuple: (class_names, train_counts, val_counts, test_counts, total_counts)
    """
    # Sort classes alphabetically for consistent ordering
    sorted_classes = sorted(folder_counts.items())
    
    class_names = []
    train_counts = []
    val_counts = []
    test_counts = []
    total_counts = []
    
    for class_name, total_count in sorted_classes:
        train, val, test = calculate_split_counts(total_count)
        
        class_names.append(class_name)
        train_counts.append(train)
        val_counts.append(val)
        test_counts.append(test)
        total_counts.append(total_count)
    
    return class_names, train_counts, val_counts, test_counts, total_counts

def plot_split_distribution(class_names, train_counts, val_counts, test_counts, total_counts):
    """
    Create a stacked bar plot showing train-validation-test split for each class.
    
    Args:
        class_names (list): List of class names
        train_counts (list): List of training set counts
        val_counts (list): List of validation set counts
        test_counts (list): List of test set counts
        total_counts (list): List of total counts per class
    """
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Colors for different splits
    train_color = '#2E8B57'  # Sea Green
    val_color = '#FF8C00'    # Dark Orange
    test_color = '#4169E1'   # Royal Blue
    
    x_pos = np.arange(len(class_names))
    bar_width = 0.6
    
    # First subplot: Stacked bar chart for splits
    bars1 = ax1.bar(x_pos, train_counts, bar_width, label='Train (56%)', 
                    color=train_color, alpha=0.8)
    bars2 = ax1.bar(x_pos, val_counts, bar_width, bottom=train_counts, 
                    label='Validation (14%)', color=val_color, alpha=0.8)
    bars3 = ax1.bar(x_pos, test_counts, bar_width, 
                    bottom=np.array(train_counts) + np.array(val_counts), 
                    label='Test (30%)', color=test_color, alpha=0.8)
    
    # Customize first subplot
    ax1.set_title('UCF Crime Dataset - Train/Validation/Test Split (56-14-30)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Class Names', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Videos', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total count labels on top of bars
    for i, total in enumerate(total_counts):
        ax1.text(i, total + 1, str(total), ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # Second subplot: Total videos per class
    bars_total = ax2.bar(x_pos, total_counts, bar_width, 
                        color='lightblue', edgecolor='navy', alpha=0.7)
    
    # Customize second subplot
    ax2.set_title('Total Videos per Class', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Class Names', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Number of Videos', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of each bar in second subplot
    for bar, count in zip(bars_total, total_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def print_split_statistics(class_names, train_counts, val_counts, test_counts, total_counts):
    """
    Print detailed statistics about the dataset split.
    
    Args:
        class_names (list): List of class names
        train_counts (list): List of training set counts
        val_counts (list): List of validation set counts
        test_counts (list): List of test set counts
        total_counts (list): List of total counts per class
    """
    print("\n" + "="*80)
    print("UCF CRIME DATASET - TRAIN/VALIDATION/TEST SPLIT STATISTICS (56-14-30)")
    print("="*80)
    
    # Overall statistics
    total_videos = sum(total_counts)
    total_train = sum(train_counts)
    total_val = sum(val_counts)
    total_test = sum(test_counts)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total Classes: {len(class_names)}")
    print(f"Total Videos: {total_videos}")
    print(f"Train Videos: {total_train} ({total_train/total_videos*100:.1f}%)")
    print(f"Validation Videos: {total_val} ({total_val/total_videos*100:.1f}%)")
    print(f"Test Videos: {total_test} ({total_test/total_videos*100:.1f}%)")
    
    print(f"\nCLASS-WISE BREAKDOWN:")
    print(f"{'Class':<35} {'Total':<8} {'Train':<8} {'Val':<6} {'Test':<6}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<35} {total_counts[i]:<8} {train_counts[i]:<8} "
              f"{val_counts[i]:<6} {test_counts[i]:<6}")
    
    print("-" * 70)
    print(f"{'TOTAL':<35} {total_videos:<8} {total_train:<8} {total_val:<6} {total_test:<6}")
    
    # Additional statistics
    avg_videos_per_class = total_videos / len(class_names)
    max_videos_class = class_names[total_counts.index(max(total_counts))]
    min_videos_class = class_names[total_counts.index(min(total_counts))]
    
    print(f"\nADDITIONAL STATISTICS:")
    print(f"Average videos per class: {avg_videos_per_class:.1f}")
    print(f"Class with most videos: {max_videos_class} ({max(total_counts)} videos)")
    print(f"Class with least videos: {min_videos_class} ({min(total_counts)} videos)")
    print(f"Standard deviation: {np.std(total_counts):.1f}")

def create_split_summary_table():
    """
    Create a summary table showing the split ratios.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['Split', 'Ratio', 'Percentage'],
        ['Training', '56%', '56.0%'],
        ['Validation', '14%', '14.0%'],
        ['Test', '30%', '30.0%'],
        ['Total', '100%', '100.0%']
    ]
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code the rows
    colors = ['#2E8B57', '#FF8C00', '#4169E1', '#gray']
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i < 4:  # Don't color the total row differently
                table[(i, j)].set_facecolor(colors[i-1])
                table[(i, j)].set_alpha(0.3)
    
    plt.title('Dataset Split Configuration', fontsize=14, fontweight='bold', pad=20)
    return fig

def main():
    """
    Main function to execute the dataset split plotting.
    """
    # Get current directory
    current_directory = os.getcwd()
    print(f"Analyzing UCF Crime dataset in: {current_directory}")
    
    # Count files in each folder
    folder_counts = count_files_in_folders(current_directory)
    
    if not folder_counts:
        print("No class folders found in the current directory.")
        return
    
    # Prepare split data
    class_names, train_counts, val_counts, test_counts, total_counts = prepare_split_data(folder_counts)
    
    # Print statistics
    print_split_statistics(class_names, train_counts, val_counts, test_counts, total_counts)
    
    # Create and display the main plot
    fig_main = plot_split_distribution(class_names, train_counts, val_counts, test_counts, total_counts)
    
    # Create summary table
    fig_table = create_split_summary_table()
    
    # Show plots
    plt.show()
    
    # Save plots option
    save_plots = input("\nDo you want to save the plots? (y/n): ").lower().strip()
    if save_plots == 'y':
        # Save main plot
        fig_main.savefig("ucf_crime_train_val_test_split.png", dpi=300, bbox_inches='tight')
        print("Main plot saved as: ucf_crime_train_val_test_split.png")
        
        # Save summary table
        fig_table.savefig("split_configuration_table.png", dpi=300, bbox_inches='tight')
        print("Summary table saved as: split_configuration_table.png")
        
        # Save statistics to text file
        with open("split_statistics.txt", "w") as f:
            f.write("UCF CRIME DATASET - TRAIN/VALIDATION/TEST SPLIT STATISTICS (56-14-30)\n")
            f.write("="*80 + "\n\n")
            
            total_videos = sum(total_counts)
            total_train = sum(train_counts)
            total_val = sum(val_counts)
            total_test = sum(test_counts)
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"Total Classes: {len(class_names)}\n")
            f.write(f"Total Videos: {total_videos}\n")
            f.write(f"Train Videos: {total_train} ({total_train/total_videos*100:.1f}%)\n")
            f.write(f"Validation Videos: {total_val} ({total_val/total_videos*100:.1f}%)\n")
            f.write(f"Test Videos: {total_test} ({total_test/total_videos*100:.1f}%)\n\n")
            
            f.write("CLASS-WISE BREAKDOWN:\n")
            f.write(f"{'Class':<35} {'Total':<8} {'Train':<8} {'Val':<6} {'Test':<6}\n")
            f.write("-" * 70 + "\n")
            
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:<35} {total_counts[i]:<8} {train_counts[i]:<8} "
                       f"{val_counts[i]:<6} {test_counts[i]:<6}\n")
        
        print("Statistics saved as: split_statistics.txt")

if __name__ == "__main__":
    main()
