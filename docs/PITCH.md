# Two-minute pitch (about 280 words)

**The problem (20 s).** A hostel mess or apartment block in Bengaluru throws wet waste, plastic, paper and foil
into the nearest bin. One food-soiled bag in the dry-waste bin can send the whole load to landfill. The person
standing at the bin has two seconds to decide and no one to ask.

**What we built (30 s).** BinSense is a decision box. Show it an item and it answers typed questions, offline, on a
Raspberry Pi 4: which category, which bin, is it recyclable, and how sure it is. It runs a pretrained image model
through ExecuTorch with the XNNPACK delegate. No cloud, no internet, no data leaves the building.

**Why it is different (30 s).** Most classifiers are trained for one fixed list of labels. Ours takes its labels from
a text file, so a new site with a different bin scheme needs a new file, not a new model. It returns several labels at
once, for example "paper, recyclable, dry waste". And when it is not sure, it says so and sends the item to a person
instead of guessing.

**What we measured (30 s).** *[Read these from results/SUMMARY.md on the Pi.]* On the organiser images: __ of __
correct. Inference time per image: __ ms on the Pi 4 at __ threads. Model size: __ MB, peak memory: __ MB.
Without the XNNPACK delegate the same model is roughly a hundred times slower, which is why delegation matters.
We also show what did not work: INT8 broke our best model, so it ships in FP32.

**Who benefits (10 s).** Housekeeping staff and residents get an instant answer at the bin; the waste contractor gets
cleaner dry waste and a log of every decision.

**Honest limits.** Ten to twenty images is a tiny test, and lighting and soiling in the field will be harder.
This is a proof of concept; production validation is still needed.

---

## If a judge asks

- **"Did you train it?"** The backbone is a public pretrained model. We also show a fine-tuned version and say which is which.
- **"Why not INT8?"** We tried. It works for MobileNetV2 (2.5 MB, about 5 points lower) and breaks MobileCLIP. Both results are in the table.
- **"Does it use KleidiAI?"** Not on a Pi 4: those kernels need newer CPU features. We use XNNPACK's Neon kernels.
- **"How do you know it is not overfitted?"** We found our own test split was optimistic (near-duplicate photos) and say so in the README.
