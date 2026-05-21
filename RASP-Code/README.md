# RASP

<p align="center">
  <strong>Revisiting 3D Anamorphic Art for Shadow-Guided Packing of Irregular Objects</strong><br/>
  Official codebase for <strong>CVPR 2025</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#running-the-code">Running the Code</a> •
  <a href="#citation">Citation</a>
</p>


## Quickstart

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate pytorch3d
```

### 2. Run arrangement optimization

```bash
cd arrangement
bash train.sh
```

Default script:

```bash
python3 train.py --config config.twoView --save_mode new --device 0
```

### 3. Run texturing optimization

```bash
cd ../texturing
bash train.sh
```

Default script:

```bash
python3 train.py --config config.config1 --save_mode new --device 0
```

## Running the Code

### Hardware / device notes
- `--device` should be a CUDA GPU index (e.g., `0`, `1`).
- The current training entrypoints are configured for GPU-first execution.

### Key configs to edit
- `arrangement/config/twoView.py`
  - `TARGET_IMAGES`
  - `SOURCE_OBJECTS_PATH`
  - `SOURCE_SCALES`
  - camera setup (`CAMERA_TRANSFORMS`, `CAMERA_TYPES`)
  - optimization settings (`NUM_EPOCHS`, weights, LR)
- `texturing/config/config1.py`
  - `TARGET_IMAGES`
  - `SOURCE_OBJECTS` (points to `final.obj` from arrangement)
  - camera + optimization settings

### Typical outputs
- `results/runs_XXXX/metadata/`: copied config, source/target snapshots
- `results/runs_XXXX/silhouettes/`: periodic rendered silhouettes
- `results/runs_XXXX/report/`: losses (`*.csv`, `*.html`), structures (`*.obj`), and analysis visualizations
- `progressive.gif`: optimization progression over iterations

## Reproducing the Included Demo

The repository already contains sample results in:
- `arrangement/results/runs_0001/`
- `texturing/results/runs_0001/`

To regenerate with current configs, rerun both stages (`arrangement` then `texturing`) and compare generated assets under new `runs_XXXX` directories.

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{debnath2025rasp,
  title={RASP: revisiting 3D anamorphic art for shadow-guided packing of irregular objects},
  author={Debnath, Soumyaratna and Tiwari, Ashish and Sadekar, Kaustubh and Raman, Shanmuganathan},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5849--5858},
  year={2025}
}
```

