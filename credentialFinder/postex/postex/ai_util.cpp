#pragma once
#include <Windows.h>

#include <cctype>
#include <vector>
#include <string>
#include <sstream>

#include <iostream>

#include "ai_util.h"

/* MIN/MAX per model architecture */
const int MIN_LENGTH = 7;
const int MAX_LENGTH = 32;

/**
 * @brief Splits the input buffer into words separated by whitespace.
 *
 * This function converts the provided byte buffer (up to the specified length)
 * into a string and splits it into individual words based on whitespace characters.
 * The resulting words are returned in a vector, excluding any empty strings that
 * may result from consecutive or leading/trailing whitespace.
 *
 * @param buffer Pointer to the input buffer containing the string data.
 * @param length Maximum number of characters to process from the buffer.
 * @return A vector of strings, each representing a word split by whitespace.
 */
std::vector<std::string> SplitBufferByWhitespace(const UCHAR* buffer, size_t length) {
    
    std::vector<std::string> output;

    // Convert input to string
    std::string input((const char*) buffer, length);

    // Convert string to stream
    std::stringstream ss(input);
    std::string word;

    while (ss >> word) {
        output.push_back(word);
    }

    return output;
}

/**
 * @brief Creates an unordered map that maps each character from the given string to a unique integer index.
 *
 * The characters are grabbed from a predefined string, which is assumed to have been used to train a model.
 * The order of characters in this string must be identical to their corresponding indices in the returned map.
 *
 * This function takes into account padding and unknown character cases, where 0 and 1 are reserved for these purposes respectively.
 *
 * @return An unordered map that maps each character from the given string to a unique integer index.
 */
std::unordered_map<char, int> GenerateChars2Idx() {
    // Grabbed these chars from Python, order must be identical to the order used to train the model
    std::string chars = "\t\n\x0b\x0c\r !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";

    std::unordered_map<char, int> char2idx;
    // Create char2idx dictionary
    char idx = 2; // 0 is reserved for padding; 1 is reserved for unknown

    for (const auto& ch : chars) {
        if (char2idx.find(ch) == char2idx.end()) {
            char2idx[ch] = idx++;
        }
    }

    return char2idx;
}

/**
 * @brief Pads a string to a specified length with null characters.
 *
 * @param w   The input string to be padded.
 * @param paddedLength  The desired length of the output string.
 *
 */
void PadWord(std::string &w, size_t paddedLength) {

    while (w.size() < paddedLength) {
        w.push_back(0);
    }

}

/**
 * @brief Encode a given word into an integer sequence using a predefined character-to-index mapping.
 *
 * This function takes a string input, maps each character to its corresponding index in the `char2idx` dictionary,
 * and returns a vector of integers representing the encoded word. If a character is not found in the dictionary,
 * it defaults to the `<UNK>` (1) index. The resulting sequence is then padded with zeros up to the maximum length.
 *
 * @param w The input string to be encoded.
 * @return A vector of integers representing the encoded word.
 */
std::vector<int64_t> EncodeWord(const std::string& w) {

    std::unordered_map<char, int> char2idx = GenerateChars2Idx();

    // Get the encoded word by mapping each character to its index in char2idx dictionary
    std::vector<int64_t> encoded(w.size(), -1);
    for (size_t i = 0; i < w.size(); ++i) {
        if (char2idx.find(w[i]) != char2idx.end()) {
            encoded[i] = char2idx.at(w[i]);
        }
        else {
            encoded[i] = 1; // Default to <UNK> if not found
        }
    }

    // Pad the vector with zeros to reach maximum length
    while (encoded.size() < MAX_LENGTH) {
        encoded.push_back(0); // Pad with <PAD> if not found
    }

    return encoded;
}

/**
 * @brief Prints an encoded word, where each character in the original word is replaced
 * with its corresponding index in a pre-defined mapping.
 *
 * This function takes a string as input, encodes it using the EncodeWord function,
 * and prints the resulting vector of indices to the console. It ignores any invalid
 * indices that may be present in the encoded word.
 *
 * @param word The original string to be encoded
 */
void PrintEncodedWord(std::string word) {

    std::vector<int64_t> encodedWord = EncodeWord(word);

    // Print the encoded word
    for (const auto& val : encodedWord) {
        if (val != -1) { // Ignore invalid indices
            std::cout << val << " ";
        }
    }
}
