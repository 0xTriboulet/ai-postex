#pragma once
/* rust.h
 * C-style prototype declerations of the functions implemented in lib.rs
 */

#define STDCALL __stdcall

#pragma comment(lib, "ntdll") // needed for Rust's stdlib

 extern "C" unsigned long long STDCALL add(unsigned long long a, unsigned long long b);

 // pub extern "stdcall" fn extract_text(file_name: *const c_char, len: usize) -> *const u8 
 extern "C" const unsigned char* STDCALL extract_text(const char* file_name, size_t name_len);

 // pub extern "stdcall" fn free_extracted_text(ptr: *mut c_char)
 extern "C" void STDCALL free_extracted_text(const char* ptr);