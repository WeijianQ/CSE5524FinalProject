# CSE 5524 Final Project

This is the implementaion of CSE 5524 final project. We build our project on top of [Video-Swin](https://github.com/SwinTransformer/Video-Swin-Transformer). 

## Installation 

We follow the same installation process as Video-Swin-Transformer. Please refer to the originial [README](video_swin_README.md) for installation instructions.
After you finish the installation, install the `pytorch_lightning`.

```bash
pip install pytorch_lightning
```

## Data Processing

### Download
Download the dataset from kaggle [website](https://www.kaggle.com/competitions/nexar-collision-prediction/data). Or use the kaggle cli.

After downloading the dataset, unzip it to the data folder:
```bash
cd data
kaggle competitions download -c nexar-collision-prediction -w
unzip -o nexar-collision-prediction.zip -d ./
```
### Clip

We provide our train/val split in `data`.

First you need to clip the video to 10-second clips. Run:
```bash
python clip_video.py --all
```

### Downsample

Second, we pre-sample the videos to a frame tensor dictionary.
```bash
python extract_tensor.py --all
python extract_tensor_test.py
```

## Models

We implement four different models for collision detection:

1. **Video Swin Transformer** (`train_video_swin.py`)
   - Uses pre-trained Swin Transformer backbone
   - Can be trained with frozen or unfrozen backbone
   - Supports different frame sampling strategies

2. **3D CNN Baseline** (`train_cnn_baseline.py`)
   - Simple 3D CNN architecture
   - Uses adaptive average pooling
   - Good baseline for comparison

3. **3D CNN with Temporal Attention** (`train_cnn_temporal.py`)
   - Extends baseline CNN with temporal attention mechanism
   - Better temporal feature extraction
   - Improved performance over baseline

4. **SlowFast Network** (`train_slowfast.py`)
   - Uses pre-trained SlowFast backbone
   - Dual-pathway architecture for different temporal resolutions
   - Can be trained with frozen or unfrozen backbone

## Training

Each model can be trained using the following command:

```bash
python train_<model_name>.py [options]
```

Common options across all models:
- `--sample_choice`: Frame sampling strategy (default: 'end_biased')
  - Choices: ['uniform', 'end_biased', 'last_segment', 'random']
- `--resume`: Path to checkpoint to resume training from
- `--do_test`: Run testing instead of training
- `--test_checkpoint_path`: Path to checkpoint for testing

Model-specific options:
- Video Swin and SlowFast:
  - `--freeze`: Whether to freeze the backbone (default: True)

Example training commands:
```bash
# Train Video Swin with frozen backbone
python train_video_swin.py --sample_choice end_biased --freeze

# Train 3D CNN with temporal attention
python train_cnn_temporal.py --sample_choice end_biased

# Test SlowFast model
python train_slowfast.py --do_test --test_checkpoint_path path/to/checkpoint
```

## Evaluation

All models are evaluated using:
- Binary accuracy
- Average precision at different time-to-event (TTE) thresholds (0.5s, 1.0s, 1.5s)
- Binary cross-entropy loss

Training progress is logged using Weights & Biases (wandb) for monitoring and comparison.

## Results

The models can be compared based on:
1. Validation accuracy
2. AP@TTE scores
3. Training time and resource usage
4. Model complexity and parameter count

For detailed results and comparisons, please refer to the project report.
