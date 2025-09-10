# This script transfers the ONNX version of the bge-base-en-v1.5 model to its pytorch architecture
# Once it has been re-loaded into the pytorch architecture, coverting to an older opset is viable
# Architecture from: https://github.com/MinishLab/model2vec/blob/main/scripts/export_to_onnx.py

from model2vec.distill import distill
import torch

# Distill a Sentence Transformer model
m2v_model = distill(model_name="BAAI/bge-base-en-v1.5", pca_dims=512)

# Define the ONNX export function
def export_to_onnx(static_model, output_path, opset_version=12):
    """
    Exports a StaticModel to ONNX format.
    
    Args:
        static_model: The StaticModel instance from model2vec.distill.
        output_path: Path to save the ONNX model.
        opset_version: ONNX opset version (default: 12).
    """
    class TorchStaticModel(torch.nn.Module):
        def __init__(self, model):
            """Wrapper to convert StaticModel to a PyTorch model."""
            super().__init__()
            embeddings = torch.tensor(model.embedding, dtype=torch.float32)
            self.embedding_bag = torch.nn.EmbeddingBag.from_pretrained(embeddings, mode="mean", freeze=True)
            self.normalize = model.normalize
            self.tokenizer = model.tokenizer
            self.unk_token_id = model.unk_token_id
            self.median_token_length = model.median_token_length

        def forward(self, input_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
            embeddings = self.embedding_bag(input_ids, offsets)
            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            return embeddings

        def tokenize(self, sentences, max_length=None):
            if max_length:
                m = max_length * self.median_token_length
                sentences = [sentence[:m] for sentence in sentences]
            encodings = self.tokenizer.encode_batch(sentences, add_special_tokens=False)
            encodings_ids = [encoding.ids for encoding in encodings]
            if self.unk_token_id is not None:
                encodings_ids = [
                    [token_id for token_id in token_ids if token_id != self.unk_token_id]
                    for token_ids in encodings_ids
                ]
            if max_length:
                encodings_ids = [token_ids[:max_length] for token_ids in encodings_ids]
            offsets = torch.tensor([0] + [len(ids) for ids in encodings_ids[:-1]], dtype=torch.long).cumsum(dim=0)
            input_ids = torch.tensor(
                [token_id for token_ids in encodings_ids for token_id in token_ids],
                dtype=torch.long,
            )
            return input_ids, offsets

    # Wrap the StaticModel into TorchStaticModel
    torch_model = TorchStaticModel(static_model)

    # Prepare dummy input for exporting
    dummy_texts = ["hello", "world"]
    input_ids, offsets = torch_model.tokenize(dummy_texts)

    # Export the PyTorch model to ONNX
    torch.onnx.export(
        torch_model,
        (input_ids, offsets),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_ids", "offsets"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "num_tokens"},
            "offsets": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )
    print(f"Model successfully exported to ONNX format at {output_path}")


# Path to save the ONNX model
onnx_output_path = "model.onnx"

# Export the distilled model to ONNX with opset 12
export_to_onnx(m2v_model, onnx_output_path, opset_version=12)
