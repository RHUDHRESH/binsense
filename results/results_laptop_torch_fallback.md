| image | label | predicted | confidence | correct | multi_select | inference_ms | yes_contains_food | yes_contains_plastic | yes_contains_metal | yes_contains_paper | yes_hazardous |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Img1.png | mixed | mixed | 0.865 | yes | mixed + contains_food + contains_metal | 47.1 | 0.675 | 0.384 | 0.56 | 0.371 | 0.037 |
| Img2.png | recyclable | recyclable | 0.683 | yes | recyclable + mixed + contains_metal | 47.8 | 0.093 | 0.448 | 0.773 | 0.365 | 0.091 |
| Img3.png | mixed | mixed | 0.76 | yes | mixed | 55.6 | 0.111 | 0.345 | 0.08 | 0.463 | 0.354 |
| Img4.png | organic | organic | 0.978 | yes | organic + contains_food | 44.2 | 0.983 | 0.036 | 0.089 | 0.157 | 0.036 |
| Img5.png | recyclable | recyclable | 0.8 | yes | recyclable + contains_plastic + contains_metal + hazardous | 46.6 | 0.057 | 0.582 | 0.965 | 0.292 | 0.558 |
| Img6.png | organic | organic | 0.994 | yes | organic + contains_food | 35.4 | 0.87 | 0.05 | 0.074 | 0.347 | 0.031 |
| Img7.png | mixed | mixed | 0.604 | yes | mixed + recyclable | 48.0 | 0.087 | 0.435 | 0.125 | 0.16 | 0.037 |
| Img8.png | e-waste | e-waste | 1.0 | yes | e-waste + hazardous | 35.9 | 0.117 | 0.104 | 0.112 | 0.347 | 0.997 |
| Img9.png | recyclable | recyclable | 0.99 | yes | recyclable + contains_plastic | 34.8 | 0.001 | 0.983 | 0.099 | 0.056 | 0.062 |
| Img10.png | organic | organic | 0.991 | yes | organic + contains_food | 41.2 | 0.573 | 0.083 | 0.095 | 0.383 | 0.264 |

**Correct: 10 / 10  |  Accuracy: 100.0%**

```json
{
 "tag": "laptop_torch_fallback",
 "model": "MobileCLIP2-S0",
 "pretrained": "dfndr2b",
 "pte": "mobileclip2-s0_192_fp32.pte",
 "pte_size_mb": 45.52,
 "resolution": 192,
 "threads": 4,
 "images": 10,
 "scored": 10,
 "correct": 10,
 "accuracy": 1.0,
 "load_ms": 7040.2,
 "inference_ms": {
  "p50": 44.2,
  "p90": 48.0,
  "min": 34.8,
  "max": 55.6
 },
 "preprocess_ms": {
  "p50": 24.4,
  "p90": 43.7
 },
 "images_per_second": 14.15,
 "peak_rss_mb": 975.1,
 "machine": "AMD64",
 "cpu_count": 12,
 "temp_start": null,
 "temp_end": null,
 "clock_arm": null,
 "throttled": null
}
```