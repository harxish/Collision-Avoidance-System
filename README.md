# Collision-Avoidance-System
A Vision Based Sense and Avoid System for Small UAV's.

---

# FAST Feature Pipeline

![](images/Screenshot_2020-08-31 Sense and Avoid for Small Unmanned Aircraft Systems - 2017-Dolph_SciTech_2017-1151 pdf.png)

---

### Input:
![](images/Input.png)

### Border and Filter:
![](images/CMO.png)

### Video:
![](images/stable.gif)

---

### Files :

main.py   ->  Image and Video functions

utils.py  ->  All utility functions

---

### TODO :

- [x] Border and CMO filter
- [x] Obstacle Detection
- [ ] CLAHE contrast enhancement
- [ ] FAST Features
- [ ] Track features using LK
- [ ] Register consecutive frames
- [ ] Compute difference frame
- [ ] Compute FAST Features
- [ ] Compute bounding box
- [ ] Tracking obstacle

---

### Tried : 

- ~~CNN~~
- ~~YOLO~~
- ~~LK~~
- ~~HMM~~
- ~~Temporal Filtering~~
- ~~Adaptive Thresholding~~
