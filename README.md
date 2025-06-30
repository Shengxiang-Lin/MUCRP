# MUCRP

Code for the paper submitted to TKDE 2024 (under review).



# Dataset

* Due to file size limitations, we have not uploaded all of the data. The Amazon data can be obtained from [this website](https://jmcauley.ucsd.edu/data/amazon/), and the Douban data can be obtained from [this website](https://github.com/fengzhu1/GA-DTCDR/tree/main).

# Requirements

- Python == 3.9
- tqdm==4.64.0
- torch==1.13.0 
- numpy==1.26.4  
- matplotlib==3.5.1
- scikit-learn==0.24.2

# Run

```python
# put the datasets into the data directory
python main.py --dataset YOUR_DATASET --K 30 --opk 0.9 --lam_vl 0.7 --lam_vg 1.0 --lam_vc 0.3 
```
