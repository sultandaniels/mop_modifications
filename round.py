import re

# Open the file and read the content
with open('../data/fsim.txt', 'r') as f:
    content = f.read()

# Function to round a number
def round_number(match):
    number = float(match.group())
    return "{:.3f}".format(number)

# Use a regular expression to find numbers and replace them with rounded numbers
rounded_content = re.sub(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', round_number, content)

# Write the rounded content back to the file
with open('fsim_round.txt', 'w') as f:
    f.write(rounded_content)