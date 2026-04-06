\# ResNet-18 Analysis



\## Top 5 layers by MAC count



| Layer Name | Param Count | MACs |

|---|---:|---:|

| Conv2d: 1-1  | 9,408   | 118,013,952 |

| Conv2d: 3-1  | 36,864  | 115,605,504 |

| Conv2d: 3-4  | 36,864  | 115,605,504 |

| Conv2d: 3-7  | 36,864  | 115,605,504 |

| Conv2d: 3-10 | 36,864  | 115,605,504 |


\## Arithmetic Intensity of Most MAC-Intensive Layer
Most MAC-intensive layer: `Conv2d: 1-1`



MACs = 118,013,952



FLOPs = 2 × MACs  

= 2 × 118,013,952  

= 236,027,904



Input bytes = 1 × 3 × 224 × 224 × 4  

= 602,112



Weight bytes = 9,408 × 4  

= 37,632



Output bytes = 1 × 64 × 112 × 112 × 4  

= 3,211,264



Total bytes = 602,112 + 37,632 + 3,211,264  

= 3,851,008



Arithmetic Intensity = FLOPs / Total bytes  

= 236,027,904 / 3,851,008  

= 61.29 FLOP/byte

Note: multiple Conv2d layers in ResNet-18 tie at 115,605,504 MACs, so any four of those tied layers are valid for the remaining top-5 entries.

