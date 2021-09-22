### This program is a very simple lemmatizer, which learns a
### lemmatization function from an annotated corpus. The function is
### so basic I wouldn't even consider it machine learning: it's
### basically just a big lookup table, which maps every word form
### attested in the training data to the most common lemma associated
### with that form. At test time, the program checks if a form is in
### the lookup table, and if so, it gives the associated lemma; if the
### form is not in the lookup table, it gives the form itself as the
### lemma (identity mapping).

### The program performs training and testing in one run: it reads the
### training data, learns the lookup table and keeps it in memory,
### then reads the test data, runs the testing, and reports the
### results.

### The program takes two command line arguments, which are the paths
### to the training and test files. Both files are assumed to be
### already tokenized, in Universal Dependencies format, that is: each
### token on a separate line, each line consisting of fields separated
### by tab characters, with word form in the second field, and lemma
### in the third field. Tab characters are assumed to occur only in
### lines corresponding to tokens; other lines are ignored.

import sys
import re


### Global variables

types = {}
freq = {}
lemmas = set()

# Paths for data are read from command line
test_file = sys.argv[2]

# Counters for lemmas in the training data: word form -> lemma -> count
lemma_count = {}

# Lookup table learned from the training data: word form -> lemma
lemma_max = {}

# Variables for reporting results
training_stats = ['Wordform types', 'Wordform tokens', 'Unambiguous types', 'Unambiguous tokens', 'Ambiguous types',
                  'Ambiguous tokens', 'Ambiguous most common tokens', 'Identity tokens']
training_counts = dict.fromkeys(training_stats, 0)

test_outcomes = ['Total test items', 'Found in lookup table', 'Lookup match', 'Lookup mismatch',
                 'Not found in lookup table', 'Identity match', 'Identity mismatch']
test_counts = dict.fromkeys(test_outcomes, 0)

accuracies = {}

### Training: read training data and populate lemma counters

train_data = open(train_file, 'r')

for line in train_data:

    # Tab character identifies lines containing tokens
    if re.search('\t', line):
        # Tokens represented as tab-separated fields
        field = line.strip().split('\t')

        # Word form in second field, lemma in third field
        form = field[1]
        lemma = field[2]
        lemmas.add(lemma)

        training_counts['Wordform tokens'] += 1

        if form == lemma:
            training_counts['Identity tokens'] += 1

        try:
            freq[form] += 1
            types[form].add(lemma)

            try:
                lemma_count[form, lemma] += 1
            except KeyError:
                lemma_count[form, lemma] = 1

        except KeyError:
            freq[form] = 1
            types[form] = set()
            types[form].add(lemma)
            training_counts['Wordform types'] += 1
            lemma_count[form, lemma] = 1

for item in types:
    if len(types[item]) > 1:
        training_counts['Ambiguous types'] += 1
        training_counts['Ambiguous tokens'] += freq[item]
        most_repeated_number = 0
        for item2 in types[item]:
            if most_repeated_number < lemma_count[item, item2]:
                most_repeated_number, most_repeated = lemma_count[item, item2], item2
        lemma_max[item] = most_repeated_number, most_repeated
        training_counts['Ambiguous most common tokens'] += most_repeated_number
    elif len(types[item]) == 1:
        for item2 in types[item]:
            lemma_max[item] = 1, item2


training_counts['Unambiguous types'] = training_counts['Wordform types'] - training_counts['Ambiguous types']
training_counts['Unambiguous tokens'] = training_counts['Wordform tokens'] - training_counts['Ambiguous tokens']

### Model building and training statistics
for item in training_counts:
    print(str(item) + "\t" + str(training_counts[item]))

accuracies['Expected lookup'] = (training_counts['Unambiguous tokens']
                                  + training_counts['Ambiguous most common tokens'])/training_counts['Wordform tokens']

accuracies['Expected identity'] = training_counts['Identity tokens']/training_counts['Wordform tokens']

print('Expected lookup' + "\t" + str(accuracies['Expected lookup']))
print('Expected identity' + "\t" + str(accuracies['Expected identity']))

### Testing: read test data, and compare lemmatizer output to actual lemma

test_data = open(test_file, 'r')

for line in test_data:

    # Tab character identifies lines containing tokens
    if re.search('\t', line):
        # Tokens represented as tab-separated fields
        field = line.strip().split('\t')

        test_counts['Total test items'] += 1

        # Word form in second field, lemma in third field
        form = field[1]
        lemma = field[2]

        try:
            if lemma_max[form][1] == lemma:
                test_counts['Found in lookup table'] += 1
                test_counts['Lookup match'] += 1
            else:
                test_counts['Found in lookup table'] += 1
                test_counts['Lookup mismatch'] += 1


        except KeyError:
            test_counts['Not found in lookup table'] += 1
            if lemma != form:
                test_counts['Identity mismatch'] += 1
            else:
                test_counts['Identity match'] += 1

for item in test_counts:
    print(str(item) + "\t" + str(test_counts[item]))


accuracies['Lookup'] = test_counts['Lookup match']/test_counts['Found in lookup table']

accuracies['Identity'] = test_counts['Identity match']/test_counts['Not found in lookup table']

accuracies['Overall'] = (test_counts['Lookup match'] + test_counts['Identity match'])/test_counts['Total test items']

### Report training statistics and test results

output = open('lookup-output.txt', 'w')

output.write('Training statistics\n')

for stat in training_stats:
    output.write(stat + ': ' + str(training_counts[stat]) + '\n')

for model in ['Expected lookup', 'Expected identity']:
    output.write(model + ' accuracy: ' + str(accuracies[model]) + '\n')

output.write('Test results\n')

for outcome in test_outcomes:
    output.write(outcome + ': ' + str(test_counts[outcome]) + '\n')

for model in ['Lookup', 'Identity', 'Overall']:
    output.write(model + ' accuracy: ' + str(accuracies[model]) + '\n')

output.close
