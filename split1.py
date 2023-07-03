import random

input_file = 'nn1.txt'
train_file = 'trainset1.txt'
test_file = 'testset1.txt'
test_percentage = 0.2

# Read the lines from the input file
with open(input_file, 'r') as file:
    lines = file.readlines()

# Shuffle the lines randomly
random.shuffle(lines)

# Calculate the number of lines for the test set
test_size = int(len(lines) * test_percentage)

# Split the lines into train and test sets
train_lines = lines[test_size:]
test_lines = lines[:test_size]

# Write the train set to the train file
with open(train_file, 'w') as file:
    file.writelines(train_lines)

# Write the test set to the test file
with open(test_file, 'w') as file:
    file.writelines(test_lines)
