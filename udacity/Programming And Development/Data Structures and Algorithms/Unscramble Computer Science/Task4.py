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
TASK 4:
The telephone company want to identify numbers that might be doing
telephone marketing. 
These are numbers that make outgoing calls but never send texts,
receive texts or receive incoming calls.
Print a list of these numbers in lexicographic order with no duplicates.
The output format should be:
<list of telemarketer numbers>
"""

# Create a set for numbers that make outgoing calls
outgoing_calls = {call[0] for call in calls}

# Create sets for numbers that receive calls, send texts, or receive texts
receiving_calls = {call[1] for call in calls}
sending_texts = {text[0] for text in texts}
receiving_texts = {text[1] for text in texts}

# Potential telemarketers: outgoing_calls but not in receiving_calls, sending_texts, or receiving_texts
telemarketers = outgoing_calls - (receiving_calls | sending_texts | receiving_texts)

# Sort the set of telemarketers lexicographically
sorted_telemarketers = sorted(telemarketers)

print("These numbers could be telemarketers: ")
for number in sorted_telemarketers:
    print(number)
