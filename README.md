# SHARP: Segmentation of Hands and Arms by  Range using Pseudo-Depth for Enhanced Egocentric 3D Hand Pose Estimation and Action Recognition

<!-- ![SHARP](./images/icpr_qualitative.png) -->
<img src="./images/icpr_qualitative.png" width="1000">

# Table of Contents
- [Abstract](#abstract)
- [Usage](#usage)
  - [Hand Pose Inference with *SHARP*](#subsection-a)
  - [Action Recognition Inference](#subsection-b)
- [Weights](#weights)
  - [*SHARP* Hand Pose Model Weights](#subsection-a)
  - [Action Recognition Model Weights](#subsection-b)




# Abstract

>*Hand pose represents key information for action recognition
in the egocentric perspective, where the user is interacting with
objects. We propose to improve egocentric 3D hand pose estimation
based on RGB frames only by using pseudo-depth images. Incorporating
state-of-the-art single RGB image depth estimation techniques, we
generate pseudo-depth representations of the frames and use distance
knowledge to segment irrelevant parts of the scene. The resulting depth
maps are then used as segmentation masks for the RGB frames. Experimental
results on H2O Dataset confirm the high accuracy of the
estimated pose with our method in an action recognition task. The 3D
hand pose, together with information from object detection, is processed
by a transformer-based action recognition network, resulting in an accuracy
of 91.73%, outperforming all state-of-the-art methods. Estimations
of 3D hand pose result in competitive performance with existing methods
with a mean pose error of 28.66 mm. This method opens up new
possibilities for employing distance information in egocentric 3D hand
pose estimation without relying on depth sensors.*



# Usage:

## Hand Pose Inference with *SHARP*

 - Download *H2O Dataset* from project page: https://taeinkwon.com/projects/h2o/
 - Install *DPT-Hybrid* from: https://huggingface.co/docs/transformers/model_doc/dpt 
 - For using *SHARP* generate pseudo-depth images running script:

```bash
python generate_depth_h2o.py
```
 
 - For not using *SHARP* set *config_h2o_3D_test.yaml*:

```yaml
use_depth: False
```
 - For using SHARP with oracle ground truth data set config_h2o_3D_test.yaml:
```yaml
use_depth: True
depth_img_type: 'gt'
```

 - Setup config file for testing *config_h2o_3D_test.yaml* with a path to downloaded weights and path to dataset
 - Run command:
```bash
python test_pose.py -c cfgs/config_h2o_3d_test.yaml
```

## Action Recognition Inference

To run inference using ground truth poses set up the config file *ar_inference.yaml*:

```yaml
hand_pose_type: 'gt_hand_pose'
load_checkpoint: True
checkpoint_path: 'your path to downloaded checkpoint'
```
>To get results on test subset, the script generates .zip file that has to be submitted to codelab challange: https://codalab.lisn.upsaclay.fr/competitions/4820 

To rget *SHARP* hand pose estimations:

```bash
python get_3D_pose.py --path "path to dataset" --imgs_path "path to imgs in dataset" --use_depth True --device 1 --load_model True --load_model_path "path to SHARP weights"
```

Re-run inference with *SHARP* weights

```yaml
hand_pose_type: 'hand_pose_3d_own_masked'
load_checkpoint: True
checkpoint_path: 'your path to downloaded checkpoint'
```

# Weights

### SHARP Hand Pose Model Weights:
>[*SHARP* 3D Hand Pose Estimation Model Weights](https://cloud.cvl.tuwien.ac.at/s/xQZs8JiaDnqcL5d)

>[*SHARP* 3D Hand Pose Estimation with Oracle Grund Truth Model Weights](https://cloud.cvl.tuwien.ac.at/s/WbE7eaf2fzfaSNe)

>[*None-SHARP* 3D Hand Pose Estimation Model Weights](https://cloud.cvl.tuwien.ac.at/s/dyzAY3swx3HjWBs)

### Action Recognition Model Weights:
>[Action Recognition with *SHARP* hand pose](https://cloud.cvl.tuwien.ac.at/s/pXBqjMRz2Qk4zG8)

