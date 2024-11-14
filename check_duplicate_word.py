import argparse
from collections import defaultdict


# Function to read the file and track duplicates by line
def main(file_path):
    word_lines = defaultdict(list)  # Dictionary to store line numbers for each word
    duplicate_word = dict()  # Dictionary to store results for duplicates
    line_result = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, 1):
            line = line.strip()
            if line:
                khmer_word, english_write = line.split('=')  # Extract the word before "="
                line_result.append(english_write)  # Append the result to the line
                if khmer_word in word_lines: # check is the word have saved before(if true add to duplicate_word)
                    duplicate_word[khmer_word] = True
                word_lines[khmer_word].append(i)  # Append line number to the word_lines entry


    # Identify duplicates and format output for each
    for word, _ in duplicate_word.items():
        print(f"The word {word} is duplicate on line {word_lines[word]}")
        for no_line in word_lines[word]:
            print(f"{no_line}-------- {line_result[no_line-1]}")


# Path to your file
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process a file path.")
    
    # Define a --path argument to take in the file path
    parser.add_argument('--path', type=str, required=True, help="Path to the file.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main function with the provided file path
    main(args.path)

"""
    example of using command:
        python check_duplicate_word.py --path file_path.text
"""
