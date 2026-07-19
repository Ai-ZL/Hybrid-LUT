# Hybrid-LUT
[ECCV 2026] Hybrid-LUT: Channel-Aware Hybrid Lookup Table and Filtering for Efficient Image Denoising

## Usage
### Dependency
```
python=3.8.16
torch=1.12.1
tensorboard=2.12.0
scipy=1.10.1
tqdm=4.65.0
opencv-python=4.7.0.72
```

### Training
```
python train_y.py
```
### Transfer to LUTs
```
python transfer.py
```

### Testing
```
python test_y.py
```

### Finetune (for real-world dataset)
```
python finetune.py
```

## Acknowledgement
Our code is based upon [HKLUT](https://github.com/jasonli0707/hklut) and [DnLUT](https://github.com/Stephen0808/DnLUT). Thanks for these awesome codes!
