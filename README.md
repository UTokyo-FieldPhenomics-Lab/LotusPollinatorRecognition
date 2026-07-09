# insect-recognition
## Domain-Informed Deep Learning for Robust Pollinator Recognition in Lotus Flowers 

## Quick Start

1. Clone the repository.

2. Download the example images (see `examples/README.md`).

3. Place the trained models in

models/
    flower_detector.pt
    insect_detector.pt

4. Run

python run_example.py \
    --input examples/input \
    --output examples/output \
    --flower_model models/flower_detector.pt \
    --insect_model models/insect_detector.pt
