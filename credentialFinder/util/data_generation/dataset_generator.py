"""
Generate a synthetic dataset of passwords and words.

This script reads in two text files: `random_passwords.txt` containing a list of unique passwords,
and `random_words.txt` containing a list of words. It then samples a specified number of passwords
and words, and combines them into a single dataset where each password is assigned a value of 1,
and each word is assigned a value of 0.

The dataset is then saved to a CSV file named `dataset.csv`.

Usage:
    python dataset_generator.py
"""

# passwords sourced from:    https://crackstation.net/files/crackstation-human-only.txt.gz
# random words sourced from: https://www.kaggle.com/datasets/ffatty/plain-text-wikipedia-simpleenglish; 

import random
import csv

RANDOM_PASSWORDS = "random_passwords.txt"
RANDOM_WORDS = "random_words.txt"

OUTPUT_CSV = "dataset.csv"


def create_dataset(passwords_file: str, words_file: str, output_file: str, sample_size: int = 2_000_000):
    """
    Create a synthetic dataset of passwords and words.

    Parameters:
        passwords_file (str): Path to the file containing the list of unique passwords.
        words_file (str): Path to the file containing the list of words.
        output_file (str): Path to the CSV file where the dataset will be saved.
        sample_size (int, optional): Number of each passwords and words to sample. Defaults to 2_000_000.

    Returns:
        None
    """
    
    print(f'[i] Reading passwords...')

    with open(passwords_file, 'r', encoding='ansi') as f:
        passwords = f.read().splitlines()
        
    passwords = [pwd for pwd in passwords if 7 <= len(pwd) <= 32]
    
    print(f'[i] Reading words...')

    with open(words_file, 'r', encoding='ansi') as f:
        words = f.read().splitlines()
    
    print(f'[i] Parsing words from each line...')

    # Get word from each line
    all_words = [word for line in words for word in line.split() if 7 <= len(word) <= 32]
    
    print(f'[i] Sampling passwords...')
    sampled_passwords = random.sample(passwords, sample_size)
    
    print(f'[i] Sampling words...')
    sampled_words = random.sample(all_words, sample_size)
    
    print(f'[i] Populating dataset with passwords...')
    dataset = [(password, 1) for password in sampled_passwords]
    
    print(f'[i] Populating dataset with words...')
    dataset += [(word, 0) for word in sampled_words]
    
    print(f'[i] Writing dataset to csv...')
    with open(output_file, 'w', newline='', encoding='ansi') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['word','is_password'])  # write header
        writer.writerows(dataset)
    
if __name__ == '__main__':
    print(f'[i] Creating dataset...')
    
    create_dataset(RANDOM_PASSWORDS, RANDOM_WORDS, OUTPUT_CSV)
    
    print(f'[i] Dataset {OUTPUT_CSV} created!')

