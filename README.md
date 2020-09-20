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
