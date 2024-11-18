"""
Read file into texts and calls.
It's ok if you don't understand how to read files.
"""
import csv

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

"""
TASK 1:
How many different telephone numbers are there in the records? 
Print a message:
"There are <count> different telephone numbers in the records."
"""

# Use a set to store unique phone numbers
unique_numbers = set()

# Add phone numbers from texts
for text in texts:
    unique_numbers.add(text[0])  # Incoming text number
    unique_numbers.add(text[1])  # Answering text number

# Add phone numbers from calls
for call in calls:
    unique_numbers.add(call[0])  # Incoming call number
    unique_numbers.add(call[1])  # Answering call number

# Count unique phone numbers
count_unique_numbers = len(unique_numbers)

print(f"There are {count_unique_numbers} different telephone numbers in the records.")
