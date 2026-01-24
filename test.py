import matplotlib.pyplot as plt
import torch

import models as models
from pipeline_terrain import TerrainDiffusionPipeline

pipe = TerrainDiffusionPipeline.from_pretrained("./weights", torch_dtype=torch.float16)
pipe.to("cuda")
prompt = "A sentinel-2 image of montane forests and mountains in Mexico in August"
seed = 42

generator = torch.Generator("cuda").manual_seed(seed)
image, dem = pipe(
    prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator
)

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(f"""Generation for "{prompt}" with seed {seed}""", fontsize=16)
ax[0].imshow(image[0])
ax[0].set_title("Generated Image")
ax[1].imshow(dem[0])
ax[1].set_title("Generated DEM")
plt.show()
