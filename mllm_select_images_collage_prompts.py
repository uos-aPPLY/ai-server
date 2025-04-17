# mllm_select_images_prompts.py
mllm_select_images_collage_instruction = """
You are presented with one or more 4×4 collages, each containing up to 16 images. (The last collage may have fewer than 16 images.)
Each image is labeled with a red number in the upper left corner.
Treat all images across all collages as a single unified collection.
Your task is to globally compare every image and select exactly 9 image numbers from the entire collection that best satisfy the <Rule> criteria below.
Follow the <Format> instructions precisely.

<Rule>
1. Group images that share at least '75%' similarity into clusters.
2. Your overall goal is to select 9 images that are as diverse as possible and avoid redundancy.
3. If there are exactly 9 unique (non-duplicated) clusters, choose the most aesthetically pleasing image from each cluster.
4. If there are more than 9 unique clusters, select one representative image (the most aesthetically appealing) from each cluster while ensuring a diversity of categories (e.g., landscapes, portraits, food, etc.).
5. If there are fewer than 9 unique clusters, select the most aesthetically pleasing image from each cluster, and then fill the remaining slots by choosing additional diverse images from clusters that contain similar images.
6. An aesthetically pleasing image should satisfy the following:
   a. It must be in sharp focus and very clear.
   b. It should be high quality (high resolution).
   c. The composition must be well balanced.
   d. The lighting should be natural, with good contrast and appropriate exposure.
   e. The overall color scheme should be harmonious, evoking an emotionally appealing atmosphere.
</Rule>

<Format>
Return only the numbers corresponding to the top 9 images that fit the <Rule>, separated by commas (","). No additional text.
</Format>
"""