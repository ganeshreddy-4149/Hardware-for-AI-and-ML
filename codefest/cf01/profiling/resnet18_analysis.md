# ResNet-18 Profiling Analysis

**Model**: ResNet-18  
**Input**: (1, 3, 224, 224)  
**Precision**: FP32  
**Tool**: torchinfo  

**Total Mult-Adds**: 1.81 GMACs  
**Total Parameters**: 11,689,512  

---

## Top 5 MAC-Intensive Layers

| Rank | Layer Name | Input Shape | Output Shape | Parameters | Mult-Adds |
|-----:|-----------|-------------|--------------|-----------:|----------:|
| 1 | Conv2d: 1-1  | [1, 3, 224, 224] | [1, 64, 112, 112] | 9,408  | 118,013,952 |
| 2 | Conv2d: 3-1  | [1, 64, 56, 56]  | [1, 64, 56, 56]   | 36,864 | 115,605,504 |
| 3 | Conv2d: 3-4  | [1, 64, 56, 56]  | [1, 64, 56, 56]   | 36,864 | 115,605,504 |
| 4 | Conv2d: 3-7  | [1, 64, 56, 56]  | [1, 64, 56, 56]   | 36,864 | 115,605,504 |
| 5 | Conv2d: 3-10 | [1, 64, 56, 56]  | [1, 64, 56, 56]   | 36,864 | 115,605,504 |

**Note:** Multiple Conv2d layers share identical MAC counts. Conv2d: 1-1 is highest due to large input resolution.

---

## Arithmetic Intensity (Most MAC-Intensive Layer)

**Layer:** Conv2d: 1-1  

---

### Step 1: FLOPs Calculation

| Description | Calculation | Result |
|------------|------------|-------:|
| MACs | Given | 118,013,952 |
| FLOPs | 2 × MACs | 2 × 118,013,952 = 236,027,904 |

---

### Step 2: Weight Memory

| Description | Calculation | Result |
|------------|------------|-------:|
| Number of weights | 3 × 64 × 7 × 7 | 9,408 |
| Weight bytes | 9,408 × 4 | 37,632 bytes |

---

### Step 3: Activation Memory

| Description | Calculation | Result |
|------------|------------|-------:|
| Input tensor | 1 × 3 × 224 × 224 | 150,528 elements |
| Input bytes | 150,528 × 4 | 602,112 bytes |
| Output tensor | 1 × 64 × 112 × 112 | 802,816 elements |
| Output bytes | 802,816 × 4 | 3,211,264 bytes |

---

### Step 4: Total Memory

| Description | Calculation | Result |
|------------|------------|-------:|
| Total bytes | 602,112 + 37,632 + 3,211,264 | 3,851,008 bytes |

---

### Step 5: Arithmetic Intensity

| Description | Calculation | Result |
|------------|------------|-------:|
| Formula | FLOPs / Total Bytes | — |
| Computation | 236,027,904 / 3,851,008 | — |
| **Final AI** | — | **61.29 FLOP/byte** |

---

## Observation

The first convolution layer dominates computation due to large spatial resolution (224×224).  
Later layers maintain high MAC counts due to repeated convolution operations in residual blocks.
