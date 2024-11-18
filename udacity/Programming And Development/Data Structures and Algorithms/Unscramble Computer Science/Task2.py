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
TASK 2:
Which telephone number spent the longest time on the phone
during the period? Don't forget that time spent answering a call
is also time spent on the phone.
Print a message:
"<telephone number> spent the longest time, <total time> seconds, on the phone during September 2016."
"""

# Create a dictionary to store total call duration for each number
time_spent = {}

for call in calls:
    incoming_number = call[0]
    answering_number = call[1]
    duration = int(call[3])

    if incoming_number in time_spent:
        time_spent[incoming_number] += duration
    else:
        time_spent[incoming_number] = duration

    if answering_number in time_spent:
        time_spent[answering_number] += duration
    else:
        time_spent[answering_number] = duration

# Find the phone number with the longest duration
max_time_number = max(time_spent, key=time_spent.get)
max_time_spent = time_spent[max_time_number]

print(f"{max_time_number} spent the longest time, {max_time_spent} seconds, on the phone during September 2016.")
