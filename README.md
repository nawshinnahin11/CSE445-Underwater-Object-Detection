# Underwater Aquarium Object Detection | CSE445

Computer Vision lab test at East West University (Fall 2026). This repository documents my YOLO11s solution for 7-class underwater object detection.

## Assignment
- Annotation format: COCO JSON, converted to normalized YOLO TXT labels
- Model: YOLO11s (COCO-pretrained)
- Training: 20 epochs with automatic mixed precision (AMP) on Kaggle T4
- Test split: 63 images
- Classes: fish, jellyfish, penguin, puffin, shark, starfish, stingray

## Repository development
This repository is being documented in stages. The original **code-only Kaggle notebook** is added in the final stage. Numbered `src/` files reproduce its code cell contents in order for readability and review. **They are sequential extracts, not independent standalone scripts.** The original notebook is the execution artifact and requires Kaggle inputs and a T4 GPU.

## Dataset
The dataset and test video were supplied by the course through Google Classroom and are not redistributed here. Update `DATA_ROOT` and `VIDEO_PATH` in the notebook to match your own Kaggle inputs. Expected image/box counts: train 448/3324, valid 127/909, test 63/582.

## Results
Results and demo will be documented in later stages.

## Academic context
This is a CSE445 laboratory assessment, not a claim of novel research. Do not treat the staged uploads as evidence of the original chronological development history.
