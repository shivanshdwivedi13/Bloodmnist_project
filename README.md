# Blood Cell Classification (BloodMNIST)

A convolutional neural network that classifies microscopy images of
blood cells into 8 types, trained on the BloodMNIST dataset
(~12,000 images, part of the MedMNIST collection).

## What's here
- `bloodmnist_nn.py` — the full pipeline: data loading, preprocessing,
  model definition, training loop, and evaluation
- `dataset_splits.json` — the train/validation/test split used
- `requirements.txt` — dependencies

## Approach
Built in PyTorch. [One or two lines: your architecture — how many conv
layers, pooling, what you used for regularisation — and how you trained
it: optimiser, learning rate, epochs.]

## Results
[Your test accuracy here, and anything notable — e.g. which classes the
model confused most often.]

## Running it
```bash
pip install -r requirements.txt
python bloodmnist_nn.py
```

## Notes
The 8 classes are visually similar in places, so the interesting part
was [whatever you actually found hard — class imbalance, a particular
pair the model kept confusing, overfitting on the small set].
