import onnx
import struct
import numpy as np

# The ReferenceEvaluator is an experimental feature in ONNX
from onnx.reference import ReferenceEvaluator

##############################################################################
# 1) Basic tokenization implementation
##############################################################################
def primitive_tokenize(batch_texts, max_length=32):
    """
    Tokenizes input strings using a fixed character-to-ID map.
    Pads with 0 (PAD_ID), and uses 100 (UNK_ID) for unknown characters.
    Returns flattened input IDs and corresponding offsets.
    """

    # Fixed vocab and special IDs, only partial vocab of what's supported by model
    vocab = {
        "!": 999, "\"": 1000, "#": 1001, "$": 1002, "%": 1003, "&": 1004, "'": 1005,
        "(": 1006, ")": 1007, "*": 1008, "+": 1009, ",": 1010, "-": 1011, ".": 1012,
        "/": 1013, "0": 1014, "1": 1015, "2": 1016, "3": 1017, "4": 1018, "5": 1019,
        "6": 1020, "7": 1021, "8": 1022, "9": 1023, ":": 1024, ";": 1025, "<": 1026,
        "=": 1027, ">": 1028, "?": 1029, "@": 1030, "[": 1031, "\\": 1032, "]": 1033,
        "^": 1034, "_": 1035, "`": 1036, "a": 1037, "b": 1038, "c": 1039, "d": 1040,
        "e": 1041, "f": 1042, "g": 1043, "h": 1044, "i": 1045, "j": 1046, "k": 1047,
        "l": 1048, "m": 1049, "n": 1050, "o": 1051, "p": 1052, "q": 1053, "r": 1054,
        "s": 1055, "t": 1056, "u": 1057, "v": 1058, "w": 1059, "x": 1060, "y": 1061,
        "z": 1062, "{": 1063, "|": 1064, "}": 1065, "~": 1066
    }

    PAD_ID = 0
    UNK_ID = 100

    input_ids_2d = []
    offsets = []
    flattened_input_ids = []

    offset = 0
    for text in batch_texts:
        # Tokenize per character using the fixed map
        token_ids = [vocab.get(char.lower(), UNK_ID) for char in text]

        # Truncate or pad
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [PAD_ID] * (max_length - len(token_ids))

        input_ids_2d.append(token_ids)
        flattened_input_ids.extend(token_ids)

        offsets.append(offset)
        offset += max_length

    return {
        "input_ids": np.array(flattened_input_ids, dtype=np.int64),
        "offsets": np.array(offsets, dtype=np.int64)
    }
##############################################################################
# 2) Load and prepare the ONNX model for reference-based inference
##############################################################################
def load_onnx_model(model_path: str) -> ReferenceEvaluator:
    # Load the ONNX model from disk
    onnx_model = onnx.load(model_path)
    
    # Extract opset version(s) from the model
    opset_info = onnx_model.opset_import

    for opset in opset_info:
        print(f"Domain: {opset.domain or 'ai.onnx'}, Opset Version: {opset.version}")
    
    # Initialize the ReferenceEvaluator for pure Python inference
    evaluator = ReferenceEvaluator(onnx_model)
    return evaluator

##############################################################################
# 3) A function that encodes texts into embeddings using the ONNX model
##############################################################################
def encode_texts(evaluator: ReferenceEvaluator, texts):
    # Convert input texts into the format your model expects
    inputs = primitive_tokenize(texts)

    # Run inference:
    #   run(None, inputs) means "compute all outputs" given 'inputs'.
    #   outputs is a list (or tuple) of NumPy arrays in the same order
    #   as the model's outputs.
    outputs = evaluator.run(None, inputs)

    # If your model produces multiple outputs, pick the correct index
    embeddings = outputs[0]
    return embeddings

##############################################################################
# 4) Demonstration
##############################################################################
if __name__ == "__main__":
    # Load the model from disk
    model_evaluator = load_onnx_model("../intelligence/models/model.onnx")

    # Example texts to encode
    texts = ["windows11-machi\\0xtriboulet"]

    # Compute embeddings
    embeddings = encode_texts(model_evaluator, texts)

    print("Embeddings shape:", embeddings.shape)
    
    for float_val in embeddings[0]:
        # Pack as a 32-bit float (little-endian)
        b = struct.pack('f', float_val)
        # Convert to hex, e.g. "3e 80 1a 3d" (one float = four hex bytes)
        hex_bytes = " ".join(f"{byte:02x}" for byte in b)
        print(hex_bytes)
