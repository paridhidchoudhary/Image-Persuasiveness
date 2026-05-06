---
license: cc-by-sa-4.0
language:
- en
size_categories:
- 50K<n<55K
---

### VisMin Dataset
**Training set:** VisMin has 65K samples for training vision-language models (e.g., CLIP, MLLM) for minimal-change understanding. The training set is publicly available. Follow the instructions below to download.

#### Download the Training Set
To download the VisMin training set, use the following commands:
```sh
git lfs install
git clone https://huggingface.co/datasets/mair-lab/vismin

```

You can load the dataset from Hugging Face repository [https://huggingface.co/datasets/mair-lab/vismin](https://huggingface.co/datasets/mair-lab/vismin) using the following command:
```python
from datasets import load_dataset, load_from_disk
dataset = load_dataset("https://huggingface.co/datasets/mair-lab/vismin")
```

Or optionallly, load from disk:
```python
dataset = load_from_disk("/path/to/vismin")
```

The training data annotations are provided in a CSV file with the following columns:

- `image_id`: Unique identifier for the image
- `image`: The image data
- `caption`: Caption associated with the image
- `bounding_boxes`: Bounding boxes of the object in the image
- `source_image_id`: For edited images, the ID of the original image (empty for original images)
- `category`: Category of the edit (empty for original images)

#### VisMin Benchmark
**Benchmark:** VisMin consists of four types of minimal-changes – object, attribute, count and spatial relation – between two image-captions pairs. 
The VisMin benchmark has 2,084 samples (579 objects, 294 attributes, 589 counting, 622 relations).

> **Note:** The VisMin benchmark is now available on Hugging Face dataset [https://huggingface.co/datasets/mair-lab/vismin-bench](https://huggingface.co/datasets/mair-lab/vismin-bench).   


### Usage
You can use dataset for training your models. Please refer to the [VisMin GitHub repository](https://github.com/rabiulcste/vismin) for more details.

### Citation Information
If you are using this dataset, please cite
```
 @article{vismin2024,
    title={VisMin: Visual Minimal-Change Understanding},
    author={Awal, Rabiul and Ahmadi, Saba and Zhang, Le and Agrawal, Aishwarya},
    year={2024}
}
```