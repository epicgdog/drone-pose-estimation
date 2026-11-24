# flo_model

Flo flow-matching lifter (Flo tag v6 = history/0040) behind an FMPose3DInference-shaped API.
- `model.pt`, `model_def.py`: weights and net (copied from Flo history/0040/model)
- `sampler.py`, `normalize.py`: copied from Flo checker/sampler.py and checker/datagen.py (normalize_2d)
- `shim.py`: `FloInference`, `YOLOCoco` (raw COCO-17 2D estimator)

Swap in inference.py: `from flo_model import FloInference` then
`self.api = FloInference(estimator_2d=YOLOTestCoco-replacement, seed=seed)` (use `YOLOCoco`; the h36m
conversion in YOLOTestCoco must NOT be applied, output is COCO-17 order, use COCO_BONES).
