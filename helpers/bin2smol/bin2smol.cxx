// clang bin2smol.cxx -o bin2smol.exe -lcabinet
// .\bin2smol.exe "C:\\Users\\0xtriboulet\\Desktop\\research\\code_workbench\\nextgen_postex\\intelligence\\models\\model.onnx"
#include <windows.h>
#include <compressapi.h>
#include <stdio.h>
#include <stdlib.h>

#define BUFFER_SIZE 65536 // 64 KB buffer size

void print_error_and_exit(const char *message) {
    DWORD error = GetLastError();
    fprintf(stderr, "%s (Error Code: %lu)\n", message, error);
    exit(EXIT_FAILURE);
}

/**
 * @file bin2smol.cxx
 * @brief A standalone Windows application that compresses a file using the LZMS compression algorithm
 *        and writes the compressed output to a new file with a `.smol` extension.
 *
 * This program:
 * - Accepts a single input file path as a command-line argument.
 * - Opens and reads the entire file into memory.
 * - Compresses the file content using the LZMS algorithm via the Windows Compression API.
 * - Writes the compressed output to a file with the same name but with a `.smol` extension.
 *
 * Usage:
 * @code
 * bin2smol.exe <file_path>
 * @endcode
 *
 * Example:
 * @code
 * bin2smol.exe C:\\path\\to\\onnx_model.bin
 * // Output: C:\path\to\onnx_model.bin.smol
 * @endcode
 *
 * @param argc Argument count; must be 2 (program name + file path).
 * @param argv Argument vector; argv[1] should be the input file path.
 * @return Returns EXIT_SUCCESS on success or EXIT_FAILURE on failure.
 *
 * @note This program is intended for Windows and uses WinAPI functions such as CreateFileA,
 *       ReadFile, WriteFile, and the Compression API (CreateCompressor, Compress, etc.).
 * 
 * @warning The entire input file is loaded into memory. Large files may lead to memory exhaustion.
 */
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file_path>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_path = argv[1];

    // Open the input file
    HANDLE input_file = CreateFileA(input_path, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (input_file == INVALID_HANDLE_VALUE) {
        print_error_and_exit("Failed to open input file");
    }

    // Get the input file size
    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(input_file, &file_size)) {
        CloseHandle(input_file);
        print_error_and_exit("Failed to get input file size");
    }

    // Allocate memory for the input file
    BYTE *input_buffer = (BYTE *)malloc((size_t)file_size.QuadPart);
    if (!input_buffer) {
        CloseHandle(input_file);
        fprintf(stderr, "Failed to allocate memory for input file\n");
        return EXIT_FAILURE;
    }

    // Read the file into memory
    DWORD bytes_read;
    if (!ReadFile(input_file, input_buffer, (DWORD)file_size.QuadPart, &bytes_read, NULL) || bytes_read != file_size.QuadPart) {
        CloseHandle(input_file);
        free(input_buffer);
        print_error_and_exit("Failed to read input file");
    }
    CloseHandle(input_file);

    // Initialize the compressor
    COMPRESSOR_HANDLE compressor;
    if (!CreateCompressor(COMPRESS_ALGORITHM_LZMS, NULL, &compressor)) {
        free(input_buffer);
        print_error_and_exit("Failed to create compressor");
    }

    // Compress the data
    SIZE_T compressed_size = 0;
    if (!Compress(compressor, input_buffer, (SIZE_T)file_size.QuadPart, NULL, 0, &compressed_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            CloseCompressor(compressor);
            free(input_buffer);
            print_error_and_exit("Failed to calculate compressed size");
        }
    }

    BYTE *compressed_buffer = (BYTE *)malloc(compressed_size);
    if (!compressed_buffer) {
        CloseCompressor(compressor);
        free(input_buffer);
        fprintf(stderr, "Failed to allocate memory for compressed data\n");
        return EXIT_FAILURE;
    }

    if (!Compress(compressor, input_buffer, (SIZE_T)file_size.QuadPart, compressed_buffer, compressed_size, &compressed_size)) {
        CloseCompressor(compressor);
        free(input_buffer);
        free(compressed_buffer);
        print_error_and_exit("Failed to compress data");
    }

    // Clean up the compressor
    CloseCompressor(compressor);
    free(input_buffer);

    // Construct the output file name
    char output_path[MAX_PATH];
    snprintf(output_path, sizeof(output_path), "%s.smol", input_path);
		
    // Write the compressed data to the output file
    HANDLE output_file = CreateFileA(output_path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (output_file == INVALID_HANDLE_VALUE) {
        free(compressed_buffer);
        print_error_and_exit("Failed to create output file");
    }

    DWORD bytes_written;
    if (!WriteFile(output_file, compressed_buffer, (DWORD)compressed_size, &bytes_written, NULL) || bytes_written != compressed_size) {
        CloseHandle(output_file);
        free(compressed_buffer);
        print_error_and_exit("Failed to write compressed data to output file");
    }

    CloseHandle(output_file);
    free(compressed_buffer);

    printf("File successfully compressed and written to: %s\n", output_path);
    return EXIT_SUCCESS;
}