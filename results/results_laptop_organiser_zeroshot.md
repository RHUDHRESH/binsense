| image | label | predicted | confidence | correct | multi_select | inference_ms | yes_contains_food | yes_contains_plastic | yes_contains_metal | yes_contains_paper | yes_hazardous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Img1.png | mixed | mixed | 0.827 | yes | mixed + contains_food + contains_metal | 39.4 | 0.656 | 0.454 | 0.623 | 0.272 | 0.05 |
| Img2.png | recyclable | recyclable | 0.825 | yes | recyclable + contains_plastic + contains_metal | 40.0 | 0.073 | 0.501 | 0.733 | 0.394 | 0.083 |
| Img3.png | mixed | mixed | 0.694 | yes | mixed + recyclable | 45.8 | 0.073 | 0.32 | 0.315 | 0.299 | 0.405 |
| Img4.png | organic | organic | 0.993 | yes | organic + contains_food | 47.3 | 0.99 | 0.034 | 0.147 | 0.148 | 0.026 |
| Img5.png | recyclable | recyclable | 0.727 | yes | recyclable + mixed + contains_metal | 51.0 | 0.061 | 0.465 | 0.908 | 0.303 | 0.333 |
| Img6.png | organic | organic | 0.994 | yes | organic + contains_food | 37.1 | 0.905 | 0.064 | 0.065 | 0.32 | 0.044 |
| Img7.png | mixed | mixed | 0.528 | yes | mixed + recyclable | 49.3 | 0.124 | 0.456 | 0.14 | 0.183 | 0.041 |
| Img8.png | e-waste | e-waste | 1.0 | yes | e-waste + hazardous | 45.9 | 0.197 | 0.099 | 0.181 | 0.37 | 0.997 |
| Img9.png | recyclable | recyclable | 0.985 | yes | recyclable + contains_plastic | 46.2 | 0.002 | 0.982 | 0.081 | 0.074 | 0.05 |
| Img10.png | organic | organic | 0.991 | yes | organic + contains_food | 46.9 | 0.648 | 0.099 | 0.118 | 0.413 | 0.269 |

**Correct: 10 / 10  |  Accuracy: 100.0%**

```json
{
 "tag": "laptop_organiser_zeroshot",
 "model": "MobileCLIP2-S0",
 "pretrained": "dfndr2b",
 "pte": "mobileclip2-s0_256_fp32.pte",
 "pte_size_mb": 45.52,
 "resolution": 256,
 "threads": 4,
 "images": 10,
 "scored": 10,
 "correct": 10,
 "accuracy": 1.0,
 "load_ms": 0.2,
 "inference_ms": {
  "p50": 45.9,
  "p90": 49.3,
  "min": 37.1,
  "max": 51.0
 },
 "preprocess_ms": {
  "p50": 23.5,
  "p90": 33.2
 },
 "images_per_second": 14.42,
 "peak_rss_mb": 426.5,
 "machine": "AMD64",
 "cpu_count": 12,
 "temp_start": null,
 "temp_end": null,
 "clock_arm": null,
 "throttled": null
}
```