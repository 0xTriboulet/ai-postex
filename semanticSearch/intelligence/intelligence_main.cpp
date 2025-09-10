#define SECURITY_WIN32 // must be defined for Security.h

#include <Security.h>
#include <windows.h>
#include <stdio.h>
#include <intrin.h>

#include "pch.h"
#include "intelligence_debug.h"
#include "rust.h"
#include "macros.h"
#include "utils.h"

#include <cstdlib>

#pragma comment(lib, "ntdll")
#pragma comment(lib, "Secur32")

#include "intelligence_utils.cpp"

void PrintModelIO(const LearningModel& model) {
#ifdef _DEBUG
    wprintf(L"\n=== INPUTS ===\n");
    for (const auto& input : model.InputFeatures()) {
        auto tensor = input.as<TensorFeatureDescriptor>();
        wprintf(L"Input name: %s\n", input.Name().c_str());
        wprintf(L"  Tensor kind: %d\n", static_cast<int>(tensor.TensorKind()));
        wprintf(L"  Data type: %d\n", static_cast<int>(tensor.TensorKind())); // TensorKind enum
        wprintf(L"  Shape: [");
        for (uint32_t i = 0; i < tensor.Shape().Size(); ++i) {
            wprintf(L"%lld", tensor.Shape().GetAt(i));
            if (i + 1 < tensor.Shape().Size()) wprintf(L", ");
        }
        wprintf(L"]\n");
    }

    wprintf(L"\n=== OUTPUTS ===\n");
    for (const auto& output : model.OutputFeatures()) {
        auto tensor = output.as<TensorFeatureDescriptor>();
        wprintf(L"Output name: %s\n", output.Name().c_str());
        wprintf(L"  Tensor kind: %d\n", static_cast<int>(tensor.TensorKind()));
        wprintf(L"  Data type: %d\n", static_cast<int>(tensor.TensorKind())); // TensorKind enum
        wprintf(L"  Shape: [");
        for (uint32_t i = 0; i < tensor.Shape().Size(); ++i) {
            wprintf(L"%lld", tensor.Shape().GetAt(i));
            if (i + 1 < tensor.Shape().Size()) wprintf(L", ");
        }
        wprintf(L"]\n");
    }
#endif
}

// base embedding model from: https://github.com/MinishLab/model2vec/ and https://huggingface.co/BAAI/bge-base-en-v1.5
extern "C" unsigned char model_onnx_smol_start[];
extern "C" unsigned char model_onnx_smol_end;

BOOL IntelligenceMain(PPOSTEX_DATA postexData){

    // null-check arguments in postexData
    if(postexData == NULL || postexData->UserArgumentInfo.Buffer == NULL){
        PostexUsage();
        return FALSE;
    }

#ifdef _DEBUG

    // "usage: <threshold-probability-float> <root-search-directory> <reference-semantic-string>
    std::vector<std::string> arguments = SplitBufferByWhitespace((const unsigned char*) postexData->UserArgumentInfo.Buffer, postexData->UserArgumentInfo.Size);

    // make sure correct number of arguments were passed in
    if (arguments.size() < 3) {
        PostexUsage();
        return FALSE;
    }

    float threshold = 0.899f;

    const CHAR* filePath = arguments.at(1).c_str();

    // concatenate remaining arguments into our semantic string 
    std::string referenceSemanticString = std::accumulate(arguments.begin() + 2, arguments.end(), std::string(),
                                             [](const std::string& a, const std::string& b) {
                                                 return a.empty() ? b : a + " " + b;
                                             });
#else
    datap parser;
    BeaconDataParse(&parser, postexData->UserArgumentInfo.Buffer, postexData->UserArgumentInfo.Size);

    /* cast first arg as file path */
    CHAR* filePath     = BeaconDataExtract(&parser, NULL);
    CHAR* semanticStr  = nullptr;
    CHAR* thresholdStr = nullptr;
    float threshold;

    thresholdStr = BeaconDataExtract(&parser, NULL); // .cna makes sure we always receive something for this arg
    threshold = std::atof(thresholdStr);

    semanticStr = BeaconDataExtract(&parser, NULL); // .cna makes sure we always receive something for this arg

    BeaconPrintf(CALLBACK_OUTPUT, "Beginning Search from: %s\nThreshold similarity: %.2f%%\nReference string: %s", filePath, (threshold*100), semanticStr);

    // concatenate remaining arguments into our semantic string 
    std::string referenceSemanticString = std::string(semanticStr);
#endif

    UCHAR*  decompressed_model = NULL;
    SIZE_T  decompressed_size  = 0x0;
    SIZE_T  model_size         = model_onnx_smol_start - &model_onnx_smol_end;

    PRINT("[i] start of model is: %p\n", model_onnx_smol_start);
    PRINT("[i] end of model is: %p\n", &model_onnx_smol_end);

    // Ensure WinRT is initialized
    init_apartment();

    // Decompress model from buffer
    DecompressData(model_onnx_smol_start, model_size, &decompressed_model, &decompressed_size);

    // Load Model
    LearningModel model = LoadModelFromBuffer(decompressed_model, decompressed_size);

    // Get a pointer to the static copy of vocab from Rust code
    const UCHAR* vocab = get_vocab();

    // Tokenize the input
    VocabType vocab_parsed = ParseVocabFromBuffer(vocab);

    PrintModelIO(model);

    // SemanticSearchDirectoryContents(LearningModel model, std::string reference_semantic_string, float threshold, const CHAR* folderPath)
    return SemanticSearchDirectoryContents(model, referenceSemanticString, vocab_parsed, threshold, filePath);
}