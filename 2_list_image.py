import os
import csv


def save_image_names_to_csv(out_folder, csv_out_path):
    # Get the names of all image files in the folder.
    image_names = [f for f in os.listdir(out_folder) if f.lower().endswith(".jpg")]
    image_names_without_extension = [os.path.splitext(f)[0] for f in image_names]

    # Read existing content from the CSV file.
    existing_data = []
    if os.path.isfile(csv_out_path):
        with open(csv_out_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            existing_data = list(reader)

    # Update the first column of the data with new image names.
    if existing_data and len(existing_data) > 0:
        header = existing_data[0]
    else:
        header = ["Image_Name"]
        existing_data.append(header)

    # Clear existing image names column and retain other columns
    updated_data = [header] + [[name] + row[1:] for name, row in zip(image_names_without_extension, existing_data[1:])]

    # If there are more new images than existing rows, append additional rows
    for name in image_names_without_extension[len(existing_data) - 1:]:
        updated_data.append([name])

    # Write the updated content back to the CSV file.
    with open(csv_out_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_data)


# Define the input and output folders
output_folder = './process_dataset'
csv_output_path = './pepper_seed.csv'

# Save the image names to a CSV file
save_image_names_to_csv(output_folder, csv_output_path)

print(f"Saved image names to {csv_output_path}.")
