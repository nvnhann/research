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
TASK 3:
(Part A):
Find all of the area codes and prefixes called by people in Bangalore.
Print the codes in lexicographic order with no duplicates.
The output format should be:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
 
(Part B):
What percentage of calls from fixed lines in Bangalore are made 
to fixed lines also in Bangalore? 
Print the answer as a percentage with 2 decimal digits.
"""

# Part A
bangalore_prefix = "(080)"
area_codes_called = set()

for call in calls:
    if call[0].startswith(bangalore_prefix):
        receiving_number = call[1]
        if receiving_number.startswith('('):
            end_idx = receiving_number.find(')')
            area_code = receiving_number[1:end_idx]
            area_codes_called.add(area_code)
        elif ' ' in receiving_number and (receiving_number[0] in '789'):
            mobile_prefix = receiving_number[:4]
            area_codes_called.add(mobile_prefix)
        elif receiving_number.startswith('140'):
            area_codes_called.add('140')

sorted_area_codes = sorted(area_codes_called)

print("The numbers called by people in Bangalore have codes:")
for code in sorted_area_codes:
    print(code)

# Part B
total_calls_from_bangalore = 0
calls_from_bangalore_to_bangalore = 0

for call in calls:
    if call[0].startswith(bangalore_prefix):
        total_calls_from_bangalore += 1
        if call[1].startswith(bangalore_prefix):
            calls_from_bangalore_to_bangalore += 1

percentage = (calls_from_bangalore_to_bangalore / total_calls_from_bangalore) * 100

print(f"{percentage:.2f} percent of calls from fixed lines in Bangalore are made to fixed lines also in Bangalore.")
