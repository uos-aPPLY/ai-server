# mllm_select_images_prompts.py
mllm_select_images_instruction = """
You pick the best image from a bunch of images based on the <Rule> section below. Follow the instructions in the <Format> section precisely.

<Rule>
Each image is labeled with a red number in the upper left corner.
Evaluate all the images using the following steps to ensure both high quality and diversity:

1. Group images that share at least '75%' similarity into clusters.
2. Your overall goal is to select 9 images that are as diverse as possible and avoid redundancy.
3. If there are exactly 9 unique (non-duplicated) clusters, choose the most aesthetically pleasing image from each cluster.
4. If there are more than 9 unique clusters, select one representative image (the most aesthetically appealing) from each cluster while ensuring a diversity of categories (e.g. landscapes, portraits, food, etc.).
5. If there are fewer than 9 unique clusters, select the most aesthetically pleasing image from each cluster, and then fill the remaining slots by choosing additional diverse images from clusters that contain similar images.
6. An aesthetically pleasing image should satisfy the following:
   a. It must be in sharp focus and very clear.
   b. It should be high quality (high resolution).
   c. The composition must be well balanced.
   d. The lighting should be natural, with good contrast and appropriate exposure.
   e. The overall color scheme should be harmonious, evoking an emotionally appealing atmosphere.
</Rule>

<Format>
Return only the numbers corresponding to the top 9 images that fit the <Rule>, separated by commas (",").**
</Format>
"""
