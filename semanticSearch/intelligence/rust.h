#pragma once

#define STDCALL __stdcall
/*
*   CHAR test_file[] = "C:\\Users\\0xtriboulet\\Desktop\\research\\code_workbench\\nextgen_postex\\test.txt";
    const unsigned char* file_content = extract_text(
        test_file, 
        strlen(test_file)
    );

    free_extracted_text((char *) file_content);
*/

// pub extern "stdcall" fn get_vocab() -> *const u8 {
extern "C" const unsigned char* STDCALL get_vocab();

// pub extern "stdcall" fn extract_text(file_name: *const c_char, len: usize) -> *const u8 
extern "C" const unsigned char* STDCALL extract_text(const char* file_name, size_t name_len);

// pub extern "stdcall" fn free_extracted_text(ptr: *mut c_char)
extern "C" void STDCALL free_extracted_text(char* ptr);