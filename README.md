# CIFAR-10 ResNet - current best (Stage E, epoch 4)
Test acc 0.8951 / macro-F1 0.8947 on the official CIFAR-10 test set (benchmark reuse, development-exposed; see metrics.json). Validation-selected from a 4-epoch continuation of the original 20-epoch model with full optimizer state carried.

## Use
pip install -r requirements.txt
python3 inference.py --selftest                      # verifies this copy end-to-end
python3 inference.py --images your_images_u8.npy     # (N,32,32,3) uint8 -> class names + probs

## Files
- model_best.keras - the model (full training state included)
- inference.py - preprocessing + prediction (resize 64x64, /255)
- sample_*.npy - 64 official-test samples + expected probs for --selftest
- metrics.json / provenance.md - numbers, selection rule, caveats, hashes
