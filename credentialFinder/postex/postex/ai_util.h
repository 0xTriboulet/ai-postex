#pragma once
#include <iostream>
#include <unordered_map>
#include <string>

std::vector<int64_t> EncodeWord(const std::string& w);
std::unordered_map<char, int> GenerateChars2Idx();
void PrintEncodedWord(std::string word);
void PadWord(std::string &w, size_t paddedLength);
std::vector<std::string> SplitBufferByWhitespace(const UCHAR* buffer, size_t length);