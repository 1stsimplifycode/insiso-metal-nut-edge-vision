# Dataset audit — MVTec AD metal_nut

- Total images: **335** (normal 242, anomalous 93)
- Normal:anomalous ratio: 2.6:1
- Defect categories: bent, color, flip, scratch
- Image shapes: {'700x700x3': 335}
- Mask shapes: {'700x700x1': 93}
- Mask pixel-value sets: {'(0, 255)': 93}
- Problems (missing/invalid/orphan/size mismatch): 0 []
- Exact duplicate groups (SHA-256): 0
- 256-bit dHash pairwise Hamming distance: {'min': 19, 'p1': 37.0, 'median': 82.0}
- Near-duplicate pairs (Hamming <= 4): 0

| Folder | Images |
|---|---|
| test/bent | 25 |
| test/color | 22 |
| test/flip | 23 |
| test/good | 22 |
| test/scratch | 23 |
| train/good | 220 |

| Category | mask area min | median | max (fraction of image) |
|---|---|---|---|
| bent | 0.0071 | 0.0187 | 0.0470 |
| color | 0.0044 | 0.0209 | 0.0657 |
| flip | 0.4785 | 0.4817 | 0.4872 |
| scratch | 0.0190 | 0.0554 | 0.1230 |
