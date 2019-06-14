# Model Performance Benchmarks
- EfficientNet-B0
  - Unoptimized
  - Optimized
- MobileNet-V3
  - Unoptimized
  - Optimized
- MnasNet-A1
  - Unoptimized
  - Optimized
- MnasNet-B1
  - Unoptimized
  - Optimized

All benchmarks run as per:

```
python onnx_export.py --model mobilenetv3_100
python onnx_optimize.py ./mobilenetv3_100.onnx --output mobilenetv3_100-opt.onnx
python onnx_to_caffe.py ./mobilenetv3_100.onnx --c2-prefix mobilenetv3
python onnx_to_caffe.py ./mobilenetv3_100-opt.onnx --c2-prefix mobilenetv3-opt
python caffe2_benchmark.py --c2-init ./mobilenetv3.init.pb --c2-predict ./mobilenetv3.predict.pb
python caffe2_benchmark.py --c2-init ./mobilenetv3-opt.init.pb --c2-predict ./mobilenetv3-opt.predict.pb
```

## EfficientNet-B0

### Unoptimized
```
Main run finished. Milliseconds per iter: 49.2862. Iters per second: 20.2897
Time per operator type:
        29.7378 ms.    60.5145%. Conv
        12.1785 ms.    24.7824%. Sigmoid
        3.62811 ms.    7.38297%. SpatialBN
        2.98444 ms.    6.07314%. Mul
       0.326902 ms.   0.665225%. AveragePool
       0.197317 ms.   0.401528%. FC
      0.0852877 ms.   0.173555%. Add
      0.0032607 ms. 0.00663532%. Squeeze
        49.1416 ms in Total
FLOP per operator type:
        0.76907 GFLOP.    95.2696%. Conv
      0.0269508 GFLOP.    3.33857%. SpatialBN
     0.00846444 GFLOP.    1.04855%. Mul
       0.002561 GFLOP.   0.317248%. FC
    0.000210112 GFLOP.  0.0260279%. Add
       0.807256 GFLOP in Total
Feature Memory Read per operator type:
        58.5253 MB.    43.0891%. Mul
        43.2015 MB.     31.807%. Conv
        27.2869 MB.    20.0899%. SpatialBN
        5.12912 MB.    3.77631%. FC
         1.6809 MB.    1.23756%. Add
        135.824 MB in Total
Feature Memory Written per operator type:
        33.8578 MB.    38.1965%. Mul
        26.9881 MB.    30.4465%. Conv
        26.9508 MB.    30.4044%. SpatialBN
       0.840448 MB.   0.948147%. Add
          0.004 MB. 0.00451258%. FC
        88.6412 MB in Total
Parameter Memory per operator type:
        15.8248 MB.    74.9391%. Conv
          5.124 MB.     24.265%. FC
       0.168064 MB.   0.795877%. SpatialBN
              0 MB.          0%. Add
              0 MB.          0%. Mul
        21.1168 MB in Total
```
### Optimized
```
Main run finished. Milliseconds per iter: 46.0838. Iters per second: 21.6996
Time per operator type:
         29.776 ms.     65.002%. Conv
        12.2803 ms.    26.8084%. Sigmoid
        3.15073 ms.    6.87815%. Mul
       0.328651 ms.   0.717456%. AveragePool
       0.186237 ms.   0.406563%. FC
      0.0832429 ms.   0.181722%. Add
      0.0026184 ms. 0.00571606%. Squeeze
        45.8078 ms in Total
FLOP per operator type:
        0.76907 GFLOP.    98.5601%. Conv
     0.00846444 GFLOP.    1.08476%. Mul
       0.002561 GFLOP.   0.328205%. FC
    0.000210112 GFLOP.  0.0269269%. Add
       0.780305 GFLOP in Total
Feature Memory Read per operator type:
        58.5253 MB.    53.8803%. Mul
        43.2855 MB.    39.8501%. Conv
        5.12912 MB.    4.72204%. FC
         1.6809 MB.    1.54749%. Add
        108.621 MB in Total
Feature Memory Written per operator type:
        33.8578 MB.    54.8834%. Mul
        26.9881 MB.    43.7477%. Conv
       0.840448 MB.    1.36237%. Add
          0.004 MB. 0.00648399%. FC
        61.6904 MB in Total
Parameter Memory per operator type:
        15.8248 MB.    75.5403%. Conv
          5.124 MB.    24.4597%. FC
              0 MB.          0%. Add
              0 MB.          0%. Mul
        20.9488 MB in Total
```
## MobileNet-V3

### Unoptimized
```
Main run finished. Milliseconds per iter: 24.8316. Iters per second: 40.2712
Time per operator type:
        15.9266 ms.    69.2624%. Conv
        2.36551 ms.    10.2873%. SpatialBN
        1.39102 ms.    6.04936%. Add
        1.30327 ms.    5.66773%. Div
       0.737014 ms.    3.20517%. Mul
       0.639697 ms.    2.78195%. Relu
       0.375681 ms.    1.63378%. Clip
       0.153126 ms.   0.665921%. FC
      0.0993787 ms.   0.432184%. AveragePool
      0.0032632 ms.  0.0141912%. Squeeze
        22.9946 ms in Total
FLOP per operator type:
       0.430616 GFLOP.    94.4041%. Conv
      0.0175992 GFLOP.    3.85829%. SpatialBN
       0.002561 GFLOP.   0.561449%. FC
     0.00210961 GFLOP.    0.46249%. Mul
     0.00173891 GFLOP.   0.381223%. Add
     0.00151626 GFLOP.    0.33241%. Div
              0 GFLOP.          0%. Relu
       0.456141 GFLOP in Total
Feature Memory Read per operator type:
        34.7354 MB.    36.4363%. Conv
        17.7944 MB.    18.6658%. SpatialBN
        14.5035 MB.    15.2137%. Mul
        9.25778 MB.    9.71113%. Relu
        7.84641 MB.    8.23064%. Add
        6.06516 MB.    6.36216%. Div
        5.12912 MB.    5.38029%. FC
        95.3317 MB in Total
Feature Memory Written per operator type:
        17.6246 MB.    26.7264%. Conv
        17.5992 MB.    26.6878%. SpatialBN
        9.25778 MB.    14.0387%. Relu
        8.43843 MB.    12.7962%. Mul
        6.95565 MB.    10.5477%. Add
        6.06502 MB.    9.19713%. Div
          0.004 MB. 0.00606568%. FC
        65.9447 MB in Total
Parameter Memory per operator type:
        16.6778 MB.    76.1564%. Conv
          5.124 MB.    23.3979%. FC
         0.0976 MB.   0.445674%. SpatialBN
              0 MB.          0%. Add
              0 MB.          0%. Div
              0 MB.          0%. Mul
              0 MB.          0%. Relu
        21.8994 MB in Total

```
### Optimized

```
Main run finished. Milliseconds per iter: 22.0981. Iters per second: 45.2527
Time per operator type:
         17.146 ms.    78.8965%. Conv
        1.38453 ms.    6.37084%. Add
        1.30991 ms.    6.02749%. Div
       0.685417 ms.    3.15391%. Mul
       0.532589 ms.    2.45068%. Relu
       0.418263 ms.    1.92461%. Clip
        0.15128 ms.   0.696106%. FC
       0.102065 ms.   0.469648%. AveragePool
      0.0022143 ms.   0.010189%. Squeeze
        21.7323 ms in Total
FLOP per operator type:
       0.430616 GFLOP.    98.1927%. Conv
       0.002561 GFLOP.   0.583981%. FC
     0.00210961 GFLOP.   0.481051%. Mul
     0.00173891 GFLOP.   0.396522%. Add
     0.00151626 GFLOP.    0.34575%. Div
              0 GFLOP.          0%. Relu
       0.438542 GFLOP in Total
Feature Memory Read per operator type:
        34.7842 MB.     44.833%. Conv
        14.5035 MB.    18.6934%. Mul
        9.25778 MB.    11.9323%. Relu
        7.84641 MB.    10.1132%. Add
        6.06516 MB.    7.81733%. Div
        5.12912 MB.    6.61087%. FC
        77.5861 MB in Total
Feature Memory Written per operator type:
        17.6246 MB.    36.4556%. Conv
        9.25778 MB.    19.1492%. Relu
        8.43843 MB.    17.4544%. Mul
        6.95565 MB.    14.3874%. Add
        6.06502 MB.    12.5452%. Div
          0.004 MB. 0.00827378%. FC
        48.3455 MB in Total
Parameter Memory per operator type:
        16.6778 MB.    76.4973%. Conv
          5.124 MB.    23.5027%. FC
              0 MB.          0%. Add
              0 MB.          0%. Div
              0 MB.          0%. Mul
              0 MB.          0%. Relu
        21.8018 MB in Total

```

## MnasNet-A1

### Unoptimized
```
Main run finished. Milliseconds per iter: 30.0892. Iters per second: 33.2345
Time per operator type:
        24.4656 ms.    79.0905%. Conv
        4.14958 ms.    13.4144%. SpatialBN
        1.60598 ms.    5.19169%. Relu
       0.295219 ms.    0.95436%. Mul
       0.187609 ms.   0.606486%. FC
       0.120556 ms.   0.389724%. AveragePool
        0.09036 ms.   0.292109%. Add
       0.015727 ms.   0.050841%. Sigmoid
     0.00306205 ms. 0.00989875%. Squeeze
        30.9337 ms in Total
FLOP per operator type:
       0.620598 GFLOP.    95.6434%. Conv
      0.0248873 GFLOP.     3.8355%. SpatialBN
       0.002561 GFLOP.   0.394688%. FC
    0.000597408 GFLOP.  0.0920695%. Mul
    0.000222656 GFLOP.  0.0343146%. Add
              0 GFLOP.          0%. Relu
       0.648867 GFLOP in Total
Feature Memory Read per operator type:
        35.5457 MB.    38.4109%. Conv
        25.1552 MB.    27.1829%. SpatialBN
        22.5235 MB.     24.339%. Relu
        5.12912 MB.    5.54256%. FC
        2.40586 MB.    2.59978%. Mul
        1.78125 MB.    1.92483%. Add
        92.5406 MB in Total
Feature Memory Written per operator type:
        24.9042 MB.    32.9424%. Conv
        24.8873 MB.      32.92%. SpatialBN
        22.5235 MB.    29.7932%. Relu
        2.38963 MB.    3.16092%. Mul
       0.890624 MB.    1.17809%. Add
          0.004 MB. 0.00529106%. FC
        75.5993 MB in Total
Parameter Memory per operator type:
        10.2732 MB.    66.1459%. Conv
          5.124 MB.    32.9917%. FC
       0.133952 MB.    0.86247%. SpatialBN
              0 MB.          0%. Add
              0 MB.          0%. Mul
              0 MB.          0%. Relu
        15.5312 MB in Total
```

### Optimized
```
Main run finished. Milliseconds per iter: 24.2367. Iters per second: 41.2597
Time per operator type:
        22.0547 ms.    91.1375%. Conv
        1.49096 ms.    6.16116%. Relu
       0.253417 ms.     1.0472%. Mul
        0.18506 ms.    0.76473%. FC
       0.112942 ms.   0.466717%. AveragePool
       0.086769 ms.   0.358559%. Add
      0.0127889 ms.  0.0528479%. Sigmoid
      0.0027346 ms.  0.0113003%. Squeeze
        24.1994 ms in Total
FLOP per operator type:
       0.620598 GFLOP.    99.4581%. Conv
       0.002561 GFLOP.    0.41043%. FC
    0.000597408 GFLOP.  0.0957417%. Mul
    0.000222656 GFLOP.  0.0356832%. Add
              0 GFLOP.          0%. Relu
       0.623979 GFLOP in Total
Feature Memory Read per operator type:
        35.6127 MB.    52.7968%. Conv
        22.5235 MB.    33.3917%. Relu
        5.12912 MB.    7.60406%. FC
        2.40586 MB.    3.56675%. Mul
        1.78125 MB.    2.64075%. Add
        67.4524 MB in Total
Feature Memory Written per operator type:
        24.9042 MB.    49.1092%. Conv
        22.5235 MB.    44.4145%. Relu
        2.38963 MB.    4.71216%. Mul
       0.890624 MB.    1.75624%. Add
          0.004 MB. 0.00788768%. FC
         50.712 MB in Total
Parameter Memory per operator type:
        10.2732 MB.    66.7213%. Conv
          5.124 MB.    33.2787%. FC
              0 MB.          0%. Add
              0 MB.          0%. Mul
              0 MB.          0%. Relu
        15.3972 MB in Total
```
## MnasNet-B1

### Unoptimized
```
Main run finished. Milliseconds per iter: 28.3109. Iters per second: 35.322
Time per operator type:
        29.1121 ms.    83.3081%. Conv
        4.14959 ms.    11.8746%. SpatialBN
        1.35823 ms.    3.88675%. Relu
       0.186188 ms.   0.532802%. FC
       0.116244 ms.   0.332647%. Add
       0.018641 ms.  0.0533437%. AveragePool
      0.0040904 ms.  0.0117052%. Squeeze
        34.9451 ms in Total
FLOP per operator type:
       0.626272 GFLOP.    96.2088%. Conv
      0.0218266 GFLOP.    3.35303%. SpatialBN
       0.002561 GFLOP.   0.393424%. FC
    0.000291648 GFLOP.  0.0448034%. Add
              0 GFLOP.          0%. Relu
       0.650951 GFLOP in Total
Feature Memory Read per operator type:
        34.4354 MB.    41.3788%. Conv
        22.1299 MB.    26.5921%. SpatialBN
        19.1923 MB.    23.0622%. Relu
        5.12912 MB.    6.16333%. FC
        2.33318 MB.    2.80364%. Add
        83.2199 MB in Total
Feature Memory Written per operator type:
        21.8266 MB.    34.0955%. Conv
        21.8266 MB.    34.0955%. SpatialBN
        19.1923 MB.    29.9805%. Relu
        1.16659 MB.    1.82234%. Add
          0.004 MB. 0.00624844%. FC
         64.016 MB in Total
Parameter Memory per operator type:
        12.2576 MB.    69.9104%. Conv
          5.124 MB.    29.2245%. FC
        0.15168 MB.   0.865099%. SpatialBN
              0 MB.          0%. Add
              0 MB.          0%. Relu
        17.5332 MB in Total
```

### Optimized
```
Main run finished. Milliseconds per iter: 26.6364. Iters per second: 37.5426
Time per operator type:
        24.9888 ms.    94.0962%. Conv
        1.26147 ms.    4.75011%. Relu
       0.176234 ms.   0.663619%. FC
       0.113309 ms.   0.426672%. Add
      0.0138708 ms.  0.0522311%. AveragePool
     0.00295685 ms.  0.0111341%. Squeeze
        26.5566 ms in Total
FLOP per operator type:
       0.626272 GFLOP.    99.5466%. Conv
       0.002561 GFLOP.   0.407074%. FC
    0.000291648 GFLOP.  0.0463578%. Add
              0 GFLOP.          0%. Relu
       0.629124 GFLOP in Total
Feature Memory Read per operator type:
        34.5112 MB.    56.4224%. Conv
        19.1923 MB.    31.3775%. Relu
        5.12912 MB.     8.3856%. FC
        2.33318 MB.    3.81452%. Add
        61.1658 MB in Total
Feature Memory Written per operator type:
        21.8266 MB.    51.7346%. Conv
        19.1923 MB.    45.4908%. Relu
        1.16659 MB.    2.76513%. Add
          0.004 MB. 0.00948104%. FC
        42.1895 MB in Total
Parameter Memory per operator type:
        12.2576 MB.    70.5205%. Conv
          5.124 MB.    29.4795%. FC
              0 MB.          0%. Add
              0 MB.          0%. Relu
        17.3816 MB in Total
```