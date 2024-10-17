import torch

def _transform(example_batch, processor):
    # inputs = processor([x.convert("RGB") for x in example_batch[img_key]], return_tensors="pt")
    inputs = processor([x.convert("L").convert("RGB") for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

def collate_fn(batch):
    return {"pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch])}
