use std::ffi::{c_char, CString};
use pdf_extract;
use std::fs;

#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!($($arg)*);
    };
}

static VOCAB: &[u8; 262028] = include_bytes!("../vocab.txt");
/// Resolves a pointer to the contents of vocab.txt
/// # Safety
/// This function is marked as `unsafe` because it returns a raw pointer. 
#[unsafe(no_mangle)]
pub extern "stdcall" fn get_vocab() -> *const u8 {
	    VOCAB.as_ptr()
}

/// Extracts text content from a file given its name and length.
///
/// # Safety
/// This function is marked as `unsafe` because it dereferences raw pointers. The caller must ensure:
/// - `file_name` is a valid, non-null pointer to a UTF-8 encoded string.
/// - `name_len` accurately represents the length of the string pointed to by `file_name`.
///
/// # Parameters
/// - `file_name`: A raw pointer to a UTF-8 encoded C string representing the file path.
/// - `name_len`: The length of the string in bytes.
///
/// # Returns
/// A raw pointer to a null-terminated UTF-8 encoded string (`*const u8`) containing the extracted text.
/// - On success, this points to a heap-allocated C string which must eventually be freed by the caller using
///   `CString::from_raw()` to avoid memory leaks.
/// - On failure, returns a null pointer (`std::ptr::null_mut()`).
///
/// # Behavior
/// - If the file is a PDF (`.pdf`), it uses `pdf_extract::extract_text_from_mem`.
/// - Otherwise, it uses `textract::extract`.
/// - The extracted text is trimmed to remove leading and trailing whitespace.
///
/// # Errors
/// Returns a null pointer in the following scenarios:
/// - The `file_name` pointer is null.
/// - The UTF-8 conversion of the input fails.
/// - The file does not exist or cannot be read.
/// - Text extraction fails.
/// - The result cannot be converted into a `CString`.
///
/// # Examples (C-style usage)
/// ```c
/// const char* file_name = "document.pdf";
/// const uint8_t* extracted = extract_text(file_name, strlen(file_name));
/// if (extracted != NULL) {
///     printf("Extracted text: %s\n", extracted);
///     free((void*)extracted);  // Make sure to free when done
/// }
/// ```
#[unsafe(no_mangle)]
pub extern "stdcall" fn extract_text(file_name: *const c_char, name_len: usize) -> *const u8 {
    // Ensure the filename pointer is not null
    if file_name.is_null() {
        debug_println!("Error: file_name pointer is null.");
        return std::ptr::null_mut();
    }

    // Convert the file_name pointer to a Rust string
    let file_name_slice = unsafe { std::slice::from_raw_parts(file_name as *const u8, name_len) };
    let str_file_name = match std::str::from_utf8(file_name_slice) {
        Ok(name) => name,
        Err(err) => {
            debug_println!("Error: Invalid UTF-8 string in file_name - {}", err);
            return std::ptr::null_mut();
        }
    };

    // Check if the file exists
    if let Err(err) = fs::metadata(str_file_name) {
        debug_println!("Error: File does not exist - {}", err);
        return std::ptr::null_mut();
    }

    // Check if the file has a .pdf extension
    let content = if str_file_name.to_lowercase().ends_with(".pdf") {
        debug_println!("File detected as PDF. Using pdf_extract method.");
        match std::fs::read(str_file_name) {
            Ok(bytes) => match pdf_extract::extract_text_from_mem(&bytes) {
                Ok(text) => text.trim().to_string(), // Sometimes we get white space back, so we need to trim it away
                Err(err) => {
                    debug_println!("Error: Failed to extract text from PDF - {}", err);
                    return std::ptr::null_mut();
                }
            },
            Err(err) => {
                debug_println!("Error: Failed to read the PDF file - {}", err);
                return std::ptr::null_mut();
            }
        }
    } else {
        debug_println!("File detected as non-PDF. Using textract method.");
        match textract::extract(str_file_name, None) {
            Ok(text) => text.trim().to_string(), // Sometimes we get white space back, so we need to trim it away
            Err(err) => {
                debug_println!("Error: Failed to extract text - {}", err);
                return std::ptr::null_mut();
            }
        }
    };

    debug_println!("file name: {:?}", str_file_name);
    debug_println!("content: {:?}", content);

    // Ensure the content is converted to a C-style string
    let c_string = match CString::new(content) {
        Ok(cstr) => cstr,
        Err(err) => {
            debug_println!("Error: Failed to convert content to CString - {}", err);
            return std::ptr::null_mut();
        }
    };

    let ret_addr = c_string.into_raw() as *const u8;

    debug_println!("ret_addr: {:?}", ret_addr);

    ret_addr
}

/// Frees memory previously allocated by `extract_text`.
///
/// # Safety
/// This function is marked `unsafe` because it operates on a raw pointer. The caller must ensure:
/// - `ptr` was originally returned from a call to `extract_text`.
/// - `ptr` has not already been freed or modified.
/// - `ptr` is not null (passing a null pointer will print a warning and do nothing).
///
/// # Parameters
/// - `ptr`: A mutable raw pointer to a C-style string (`*mut c_char`) allocated by `extract_text`.
///
/// # Behavior
/// - Reclaims ownership of the memory by converting the raw pointer back into a `CString`.
/// - The `CString` is dropped immediately, freeing the memory.
/// - If a null pointer is passed, the function will print a warning and return without taking any action.
///
/// # Examples (C-style usage)
/// ```c
/// const uint8_t* text = extract_text(file_name, strlen(file_name));
/// if (text != NULL) {
///     printf("Extracted text: %s\n", text);
///     free_extracted_text((char*)text); // Safe cleanup
/// }
/// ```
///
/// # Notes
/// - Always call this function to release memory allocated by `extract_text` to prevent memory leaks.
/// - Do not attempt to free the memory using `free()` from C; it must be freed via this Rust function.
///
/// # Warning
/// Double freeing or passing an invalid pointer leads to undefined behavior.
#[unsafe(no_mangle)]
pub extern "stdcall" fn free_extracted_text(ptr: *mut c_char) {
    // Ensure the pointer is not null
    if ptr.is_null() {
        eprintln!("Warning: Attempted to free a null pointer.");
        return;
    }

    unsafe {
        // Reclaim the CString from the raw pointer and automatically drop it to free memory
        let _ = CString::from_raw(ptr);
    }
}


#[cfg(test)]
mod tests {
    use std::env::temp_dir;
    use std::ffi::{c_char, CString};
    use std::fs::{remove_file, File};
    use std::io::Write;
    use rand::distr::Alphanumeric;
    use rand::Rng;
    use super::*;

    #[test]
    fn add_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_extract_text_non_existent_file() {
        // Define a non-existent file path
        let non_existent_file = "this_file_should_not_exist_123456789.txt";

        // Convert the file path to a C-style string (null-terminated)
        let file_name_cstr = CString::new(non_existent_file).expect("CString conversion failed");

        // Call the `extract_text` function
        let result_ptr = extract_text(file_name_cstr.as_ptr(), non_existent_file.len());

        // Assert that the result is null
        assert!(result_ptr.is_null(), "Expected extract_text to return null for a non-existent file.");
    }

    #[test]
    fn test_extract_text() {
        // Generate random content for the file
        let content: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(50) // Generate a string of 50 random characters
            .map(char::from)
            .collect();

        assert_eq!(content.len(), 50); // 50 characters

        // Generate a unique random file name and create a temporary file path
        let random_file_name: String = rand::rng()
            .sample_iter(&Alphanumeric)
            .take(10) // Generate a random file name of length 10
            .map(char::from)
            .collect();
        let mut temp_path = temp_dir();
        temp_path.push(format!("test_extract_text_{}.txt", random_file_name)); // Append random name to the path

        // Write the content to the temporary PDF file
        {
            let mut file = File::create(&temp_path).expect("Failed to create temp file.");
            write!(file, "{}", content).expect("Failed to write to temp file.");
        }

        // Call the `extract_text` function
        let temp_path_cstr = CString::new(temp_path.to_str().unwrap()).expect("CString conversion failed");
        let extracted_text = extract_text(temp_path_cstr.as_ptr(), temp_path.to_str().unwrap().len());

        // Cleanup: Remove temporary file
        remove_file(&temp_path).expect("Failed to remove temp file.");

        let extracted_text_str = unsafe { CString::from_raw(extracted_text as *mut c_char)}; // Convert the extracted data to a string

        let extracted_text_str = match extracted_text_str.to_str() {
            Ok(s) => s,
            Err(_) => panic!("Failed to convert extracted text to string."),
        };
        assert_eq!(extracted_text_str, content);
    }

    #[test]
    fn test_extract_text_specific_file() {
        use std::ffi::CString;

        // Define the file path and content
        let file_path = r"<test dir>";
        let expected_content = "this is a test";

        // Convert the file path and extension to C-style strings (null-terminated)
        let file_path_cstr = CString::new(file_path).expect("CString conversion for file path failed");

        // Call the `extract_text` function
        let extracted_text_ptr = extract_text(
            file_path_cstr.as_ptr(),
            file_path.len()
        );

        // Convert the result back to a Rust string
        let extracted_text_cstr = unsafe { CString::from_raw(extracted_text_ptr as *mut c_char) };

        let extracted_text = match extracted_text_cstr.to_str() {
            Ok(s) => s,
            Err(_) => panic!("Failed to convert extracted text to a valid UTF-8 string"),
        };

        // Assert that the extracted content matches the expected content
        assert_eq!(extracted_text, expected_content);
    }

}