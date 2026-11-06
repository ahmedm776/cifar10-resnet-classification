# Provenance
- Base artifact: original 20-epoch model2_manual_resnet.keras, sha256 335547fbd219ebdd81ceaabc919a3ca21e5875c92510ead618f0b4df2eacd643 (full Adam state, iterations=14080).
- Stage E: 4 continued epochs (E1-E4) from that exact state; lr 2.5e-4, RLR halved to 1.25e-4 after E2; batch 64, 704 steps/epoch; preprocessing resize 64x64 + /255; train/val split sha256 a4ded164ac7991a893303cc899078391c903d2f364ad7144c7098d2d1a95b887 / 64422713980b9882c6955e4ca5c5455e8b5faa9e146b5752ea7a521ac1d48699 (seed 42, stratified, 45k/5k).
- Sibling trajectory: per-process RNG reseed means per-step batch order is not bit-identical to any prior process; structure/data/optimizer/pipeline fidelity gates all passed; a control gate reproduced the recorded warm epoch-1 val_loss bit-exactly (0.3669273257255554).
- Selection: E4 chosen on heldout validation only (best val_loss 0.32215610); exactly one official-test evaluation afterward.
- Full audit trail (all epoch checkpoints, state, scripts, per-sample outputs): cifar10-stagee-resnet-extension-v1.zip (2 parts) in the same Drive folder, sha256 82b11415366980db6fe0725e39e18368e59b2bcc92a47865cd44eed672380039.
