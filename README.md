# Collision-Avoidance-System
A Vision Based Sense and Avoid System for Small UAV's.

### Input:
![](images/Input.png)

### Border and Filter:
![](images/CMO.png)

### Video:
![](images/stable.gif)

---

### Files :

main.py   ->  Video Processing and performing ops in order.

src/horizon.py  ->  Horizon class for predicting horizon and also keeping track of previousily detected horizons to calculate EMWA.

src/detect.py   ->  Detect small objects by comparing with previous frames and also by removing False Positives.

src/SORT.py     ->  Simple online and Realtime tracking for detected objects using Kalman filters.

---

### Citing

SORT

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }
    
---

### TODO :

- [x] Border and CMO filter
- [x] Obstacle Detection
- [ ] Tracking obstacle

---

### Tried : 

- ~~CNN~~
- ~~YOLO~~
- ~~LK~~
- ~~HMM~~
- ~~Temporal Filtering~~
- ~~Adaptive Thresholding~~
