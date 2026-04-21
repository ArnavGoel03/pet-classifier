# Pet Classifier

A convolutional neural network that classifies pet breeds from photos, built on top of the **Oxford-IIIT Pet Dataset** (37 breeds of cats and dogs, ~7,400 images).

Grew out of a high-school AI elective at Delhi Public School, R. K. Puram — the original used a private school-provided dataset; this reproducible version swaps in the public Oxford-IIIT benchmark so anyone can clone and run it.

## What it does

- Downloads the Oxford-IIIT Pet Dataset via `tensorflow_datasets`
- Transfer-learns on **MobileNetV2** pretrained on ImageNet
- Fine-tunes the top layers for 37-way breed classification
- Saves the trained model and reports test accuracy + confusion matrix

Expected test accuracy: **~92%** after ~10 epochs of head-training plus ~5 epochs of fine-tuning (a handful of minutes on CPU, seconds on GPU).

## Architecture

```
Input (224 × 224 × 3)
 → MobileNetV2 base (ImageNet weights, frozen during head-train)
 → GlobalAveragePooling2D
 → Dropout (0.3)
 → Dense (37, Softmax)
```

Two-stage training:

1. **Head training** — base frozen, train only the classification head (10 epochs).
2. **Fine-tuning** — unfreeze last ~30 layers of the base, retrain at a lower LR (5 epochs).

Optimizer: Adam · Loss: sparse categorical cross-entropy · Batch size: 32 · Image size: 224 × 224.

## Usage

```bash
pip install -r requirements.txt

python train.py          # downloads the dataset, trains, saves models/pet_classifier.keras
python evaluate.py       # test-set accuracy + per-class metrics + confusion matrix
python predict.py <path-to-image>   # predicts the breed of a single image
```

First run will download the dataset (~800MB) into `~/tensorflow_datasets/` — subsequent runs are instant.

## The 37 breeds

**Cats (12):** Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx.

**Dogs (25):** American Bulldog, American Pit Bull Terrier, Basset Hound, Beagle, Boxer, Chihuahua, English Cocker Spaniel, English Setter, German Shorthaired, Great Pyrenees, Havanese, Japanese Chin, Keeshond, Leonberger, Miniature Pinscher, Newfoundland, Pomeranian, Pug, Saint Bernard, Samoyed, Scottish Terrier, Shiba Inu, Staffordshire Bull Terrier, Wheaten Terrier, Yorkshire Terrier.

## Project structure

```
.
├── train.py          · data loading, augmentation, two-stage transfer-learning
├── evaluate.py       · test accuracy, classification report, confusion matrix
├── predict.py        · single-image inference CLI
├── requirements.txt
└── models/           · saved .keras checkpoints (gitignored)
```

## Dataset

> Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. *Cats and Dogs.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2012.
> <https://www.robots.ox.ac.uk/~vgg/data/pets/>

## License

MIT.
