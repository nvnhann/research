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
TASK 0:
What is the first record of texts and what is the last record of calls?
Print messages:
"First record of texts, <incoming number> texts <answering number> at time <time>"
"Last record of calls, <incoming number> calls <answering number> at time <time>, lasting <during> seconds"
"""

# First record of texts
first_text = texts[0]
first_text_incoming_number = first_text[0]
first_text_answering_number = first_text[1]
first_text_time = first_text[2]

# Last record of calls
last_call = calls[-1]
last_call_incoming_number = last_call[0]
last_call_answering_number = last_call[1]
last_call_time = last_call[2]
last_call_duration = last_call[3]

print(
    f"First record of texts, {first_text_incoming_number} texts {first_text_answering_number} at time {first_text_time}")
print(
    f"Last record of calls, {last_call_incoming_number} calls {last_call_answering_number} at time {last_call_time}, lasting {last_call_duration} seconds")
