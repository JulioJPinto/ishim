import csv

def sort_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Read all lines and store them in a list
        lines = list(reader)

    # Function to extract the sorting key from each line
    def sort_key(line):
        # Split the line by comma and take the second part
        return line[1]

    # Sort the lines based on the extracted key
    sorted_lines = sorted(lines, key=sort_key)

    # Write the sorted lines to a new CSV file
    with open(output_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sorted_lines)

# Specify the input and output file names
input_file = 'orcamento.csv'
output_file = 'sorted_orcamento.csv'

# Call the sort_csv function
sort_csv(input_file, output_file)
