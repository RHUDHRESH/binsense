| image | label | predicted | confidence | correct | multi_select | inference_ms |
|---|---|---|---|---|---|---|
| cardboard1.jpg | cardboard | cardboard | 0.914 | yes | cardboard + recyclable + dry_waste + paper_stream | 39.9 |
| cardboard108.jpg | cardboard | cardboard | 0.917 | yes | cardboard + recyclable + dry_waste + paper_stream | 41.5 |
| cardboard117.jpg | cardboard | cardboard | 0.92 | yes | cardboard + recyclable + dry_waste + paper_stream | 40.8 |
| cardboard126.jpg | cardboard | cardboard | 0.909 | yes | cardboard + recyclable + dry_waste + paper_stream | 40.1 |
| cardboard135.jpg | cardboard | cardboard | 0.932 | yes | cardboard + recyclable + dry_waste + paper_stream | 42.1 |
| glass1.jpg | glass | glass | 0.926 | yes | glass + recyclable + dry_waste | 40.0 |
| glass11.jpg | glass | glass | 0.919 | yes | glass + recyclable + dry_waste | 41.3 |
| glass120.jpg | glass | glass | 0.928 | yes | glass + recyclable + dry_waste | 41.4 |
| glass131.jpg | glass | glass | 0.925 | yes | glass + recyclable + dry_waste | 39.7 |
| glass142.jpg | glass | glass | 0.914 | yes | glass + recyclable + dry_waste | 40.7 |
| metal1.jpg | metal | metal | 0.915 | yes | metal + recyclable + dry_waste | 42.9 |
| metal108.jpg | metal | metal | 0.918 | yes | metal + recyclable + dry_waste | 40.8 |
| metal117.jpg | metal | metal | 0.92 | yes | metal + recyclable + dry_waste | 41.5 |
| metal126.jpg | metal | metal | 0.919 | yes | metal + recyclable + dry_waste | 41.8 |
| metal135.jpg | metal | metal | 0.898 | yes | metal + recyclable + dry_waste | 41.4 |
| paper1.jpg | paper | paper | 0.915 | yes | paper + recyclable + dry_waste + paper_stream | 40.2 |
| paper111.jpg | paper | paper | 0.922 | yes | paper + recyclable + dry_waste + paper_stream | 41.7 |
| paper124.jpg | paper | paper | 0.926 | yes | paper + recyclable + dry_waste + paper_stream | 42.3 |
| paper137.jpg | paper | paper | 0.925 | yes | paper + recyclable + dry_waste + paper_stream | 40.1 |
| paper15.jpg | paper | paper | 0.925 | yes | paper + recyclable + dry_waste + paper_stream | 43.6 |
| plastic1.jpg | plastic | plastic | 0.9 | yes | plastic + recyclable + dry_waste | 44.5 |
| plastic11.jpg | plastic | plastic | 0.826 | yes | plastic + recyclable + dry_waste | 41.5 |
| plastic120.jpg | plastic | plastic | 0.926 | yes | plastic + recyclable + dry_waste | 40.3 |
| plastic131.jpg | plastic | plastic | 0.915 | yes | plastic + recyclable + dry_waste | 42.0 |
| plastic142.jpg | plastic | plastic | 0.902 | yes | plastic + recyclable + dry_waste | 41.4 |
| trash1.jpg | trash | trash | 0.839 | yes | trash + dry_waste + landfill | 39.3 |
| trash101.jpg | trash | trash | 0.911 | yes | trash + dry_waste + landfill | 41.4 |
| trash104.jpg | trash | trash | 0.879 | yes | trash + dry_waste + landfill | 43.6 |
| trash107.jpg | trash | trash | 0.914 | yes | trash + dry_waste + landfill | 41.0 |
| trash11.jpg | trash | trash | 0.854 | yes | trash + dry_waste + landfill | 41.9 |

**Correct: 30 / 30  |  Accuracy: 100.0%**

```json
{
 "tag": "laptop_multi_demo",
 "model": "open_clip MobileCLIP2-S0 dfndr2b (image tower), fine-tuned on TrashNet",
 "pretrained": "fine-tuned",
 "pte": "own_mobileclip2_s0_256_fp32.pte",
 "pte_size_mb": 45.52,
 "resolution": 256,
 "threads": 4,
 "images": 30,
 "scored": 30,
 "correct": 30,
 "accuracy": 1.0,
 "load_ms": 0.2,
 "inference_ms": {
  "p50": 41.4,
  "p90": 42.9,
  "min": 39.3,
  "max": 44.5
 },
 "preprocess_ms": {
  "p50": 3.3,
  "p90": 3.5
 },
 "images_per_second": 22.35,
 "peak_rss_mb": 424.0,
 "machine": "AMD64",
 "cpu_count": 12,
 "temp_start": null,
 "temp_end": null,
 "clock_arm": null,
 "throttled": null
}
```