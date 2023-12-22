import csv

# Read CSV file and extract questions
with open('./data/test_filtered.csv', 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # Skip the header
    data = list(csv_reader)

# Define the output file name
output_file_name = 'test_filtered_processed.txt'

# Write numbered questions to the output file
with open(output_file_name, 'w', encoding='utf-8') as txt_file:
    for i, row in enumerate(data, start=1):
        question = row[0].strip()  # Assuming the question is in the first column
        # Check if the question ends with 'Yes' or 'No' and modify accordingly
        if question.endswith('Yes or no?'):
            question = question[:-10].strip()
        else:
            question = f"Question: {question} Answer: "
        # Write to the file with the desired format
        txt_file.write(f"question-{i:03d}\t{question}\n")
