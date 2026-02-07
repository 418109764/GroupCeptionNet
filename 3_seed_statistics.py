# Count the proportion of Germ in CSV file
import pandas as pd

# Read the CSV file
file_path = 'pepper_seed.csv'
df = pd.read_csv(file_path)

# Count the number of each category
category_counts = df['Germ'].value_counts()
total_count = category_counts.sum()

# Calculate the percentage of each category
category_percentages = (category_counts / total_count) * 100

# Create a DataFrame to display the statistics
category_stats = pd.DataFrame({
    'Count': category_counts,
    'Percentage': category_percentages.apply(lambda x: f"{x:.2f}%")
})

# Display the statistics
print(category_stats)
print(f"\nTotal count: {total_count}")