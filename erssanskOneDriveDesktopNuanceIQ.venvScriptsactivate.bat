[33mcommit 10ecac1e9d8495599c31d7f5250a9b2956bc73ea[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m)[m
Author: Nidhi Varade <sanskrutivarade@gmail.com>
Date:   Wed Jun 24 11:01:44 2026 +0530

    feat: add fine-tuned DistilBERT model (93.1% accuracy)

 .../checkpoints/checkpoint-1564/config.json        |    28 [32m+[m
 .../checkpoints/checkpoint-1564/model.safetensors  |   Bin [31m0[m -> [32m267832560[m bytes
 .../checkpoints/checkpoint-1564/optimizer.pt       |   Bin [31m0[m -> [32m535729227[m bytes
 .../checkpoints/checkpoint-1564/rng_state.pth      |   Bin [31m0[m -> [32m14645[m bytes
 .../checkpoints/checkpoint-1564/scheduler.pt       |   Bin [31m0[m -> [32m1465[m bytes
 .../checkpoints/checkpoint-1564/trainer_state.json |   172 [32m+[m
 .../checkpoints/checkpoint-1564/training_args.bin  |   Bin [31m0[m -> [32m5201[m bytes
 .../checkpoints/checkpoint-2346/config.json        |    28 [32m+[m
 .../checkpoints/checkpoint-2346/model.safetensors  |   Bin [31m0[m -> [32m267832560[m bytes
 .../checkpoints/checkpoint-2346/optimizer.pt       |   Bin [31m0[m -> [32m535729227[m bytes
 .../checkpoints/checkpoint-2346/rng_state.pth      |   Bin [31m0[m -> [32m14645[m bytes
 .../checkpoints/checkpoint-2346/scheduler.pt       |   Bin [31m0[m -> [32m1465[m bytes
 .../checkpoints/checkpoint-2346/trainer_state.json |   240 [32m+[m
 .../checkpoints/checkpoint-2346/training_args.bin  |   Bin [31m0[m -> [32m5201[m bytes
 .../checkpoints/checkpoint-782/config.json         |    28 [32m+[m
 .../checkpoints/checkpoint-782/model.safetensors   |   Bin [31m0[m -> [32m267832560[m bytes
 .../checkpoints/checkpoint-782/optimizer.pt        |   Bin [31m0[m -> [32m535729227[m bytes
 .../checkpoints/checkpoint-782/rng_state.pth       |   Bin [31m0[m -> [32m14645[m bytes
 .../checkpoints/checkpoint-782/scheduler.pt        |   Bin [31m0[m -> [32m1465[m bytes
 .../checkpoints/checkpoint-782/trainer_state.json  |   104 [32m+[m
 .../checkpoints/checkpoint-782/training_args.bin   |   Bin [31m0[m -> [32m5201[m bytes
 model/artifacts/distilbert-imdb/config.json        |    28 [32m+[m
 .../artifacts/distilbert-imdb/confusion_matrix.png |   Bin [31m0[m -> [32m35173[m bytes
 model/artifacts/distilbert-imdb/metrics.json       |    10 [32m+[m
 model/artifacts/distilbert-imdb/model.safetensors  |   Bin [31m0[m -> [32m267832560[m bytes
 model/artifacts/distilbert-imdb/roc_curve.png      |   Bin [31m0[m -> [32m49533[m bytes
 model/artifacts/distilbert-imdb/tokenizer.json     | 30686 [32m+++++++++++++++++++[m
 .../distilbert-imdb/tokenizer_config.json          |    15 [32m+[m
 model/artifacts/distilbert-imdb/training_args.bin  |   Bin [31m0[m -> [32m5201[m bytes
 29 files changed, 31339 insertions(+)
