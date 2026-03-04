from torch.utils.data import DataLoader
from meme_dataset import MemeOnlineRLDataset
from transformers import AutoProcessor

# download from https://huggingface.co/datasets/luodi-7/Eimages
REAL_ANNOTATION = "/root/data/Eimages/train_data.json"
REAL_ROOT = "/root/data/Eimages"
MODEL_PATH = "/root/model/HUMOR-COT-Qwen2.5-VL"
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)


def test_real_dataset_loading():
    dataset = MemeOnlineRLDataset(
        annotation_path=REAL_ANNOTATION, root_dir=REAL_ROOT, shuffle=False, processor=processor
    )

    # print(len(dataset))
    assert len(dataset) == 3345

    prompt, image_path, label, reference = dataset[0]
    assert "Meme Text Generation Framework" in prompt
    assert "[Comprehensive Description Section]" in reference
    assert isinstance(image_path[0], str)
    # from PIL import Image
    # image = Image.open(image_path[0])

    # outputs = processor(
    #     text=prompt,
    #     images=image,
    #     add_special_tokens=False,
    #     max_length=4096,
    #     truncation=True,
    # )


def test_collate_fn():
    dataset = MemeOnlineRLDataset(annotation_path=REAL_ANNOTATION, root_dir=REAL_ROOT, processor=processor)
    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.collate_fn)

    batch = next(iter(loader))
    assert len(batch[0]) == 4
    assert len(batch[1]) == 4
    assert len(batch[2]) == 4
    assert len(batch[3]) == 4


if __name__ == "__main__":
    test_real_dataset_loading()
    test_collate_fn()
