# update 2023/08

Our paper "Various frameworks for integrating image and video streams for spatiotemporal information learning employing 2D-3D Residual networks for human action recognition" is under review 



## Summary

This is the PyTorch code for the video-based reconition 
```

## Pre-trained models


resnet-101-kinetics.pth: --model resnet --model_depth 101 --resnet_shortcut B

that fine-tuned models on UCF-101 and HMDB-51
```

### preparing the model on datasets for video-based recognition 

### UCF-101
```
* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

##Bash code
  python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

##bash code
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```

### HMDB-51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

