# AutoChar - Version 0.9.5 is here!
https://civitai.com/models/95923/autochar-easy-character-art-creation-with-face-auto-inpaint

AutoChar Control Panel is a custom script for Stable Diffusion WebUI by Automatic1111 (1.6.0+) made to help newbies and enthusiasts alike achieve great pictures with less effort.  
Basically, it's automation of my basic SD workflow for illustrations (check 'em here: https://t.me/dreamforge_tg) and also my Bachelor graduation work, for which I got an A. I've put decent emphasis in code readability and comments for it, so I hope it will help future contributors and developers of other scripts.  

Please check my new guide for it on YouTube that explain all basic functions and pipeline: https://www.youtube.com/watch?v=jNUMHtH1U6E  
For text description of scripts' basic idea check 0.9 version tab on CivitAI page.

## Installation
**Just put script and .onnx face recognition model in your stable-diffusion-webui/scripts folder  
PLEASE, don't try to install via URL, it's not an extension, it won't be visible this way!  
Also I highly recommend to download 4x-UltraSharp Upscaler (https://mega.nz/folder/qZRBmaIY#nIG8KyWFcGNTuMX_XNbJ_g) and put in /modes/ESRGAN folder**


## How to use, in short
1. Go to your txt2img tab
2. Write prompt, select basic parameters as usual (you don't need highres fix, since it's included in the algorithm)
3. Select "AutoChar Beta 0.9" in dropdown menu Scripts in the lower part of page
4. Click "Generate" and enjoy
![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/16919ca6-1de3-4052-a2fd-6729c9e890e5)

### 0.9.5 changes:
- Fully revamped interface:
  - Info added for all crucial parameters, containing tips for usage and clarification for not-so-obvious functions
  - Upscaler choosing changed from check panels to dropdowns to reduce distraction
  - Function and slider groups divided to different blocks with clarification headers
- True img2img mode: edit existing pictures with SD upscale and automatic face&eyes inpaint
- Additional **Advanced options**!
-	Brand new **Really Advanced options** tab for brave enthusiasts willing to take complete control of AutoChar's generation pipeline and maximize their creativity
- Various fixes:
  - Fixed infamous bug with OpenCV on inpaint step
  -	Fixed inpaint only masked padding, drastically improving results on some artstyles and checkpoints
  -	Fixed High-Res Fix upscalers' list, now it shows all available upscalers as it should
  -	Styles from Styles Menu are now working properly
  -	Many small fixes in code's logic and parameters
 
![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/9eccfbaa-8a09-41c7-ab97-856e6b16c7d4)

### Comprehensive description of Advanced and Really Advanced options and tips for their usage:
-	<u>**Advanced options:**</u>
  -	<u>Quality functions:</u>
    -	**Filtering function:** sharpens and applies denoising filter to image after High-Res Fix to improve quality and reduce the number of necessary   
 img2img steps. May negatively impact desired result on "noisy" and blurry artstyles. _On by default_
    -	**Inpaint only the biggest face on the image:** does what it says, can be great to prevent undesired face detection and inpaint of background or body parts. May cause problems on images with small character head (full-height pictures and landscapes). In this case, either increase Face Recognition minimum confidence or disable this options. Also disable for pictures with two or more characters._On by default_
    -	**Lower LoRA strength for face inpaint. Helps avoid burnout with strong LORAs:** does what it says._On by default_
    -	**Use DDIM sampler for better inpaint. Will use chosen in interface otherwise:** better for detailed faces. Note that from SD WebUi's version 1.6.0+ denoising strength works differently for DMP++ samplers, so if you're disabling this option because of possible mask residue issues, consider increasing denoising strength for inpaint steps. _On by default_
    -	**Lower CFG for face inpaint. Helps avoid burning with multiple LoRAs:** does what it says. _Off by default_
- <u> Algorithm-alterting functions:</u>
  -	**Make face inpaint box larger to inpaint hair along with the face:** does what it says. It can become quite VRAM heavy, so consider lowering Scaling factor for face inpainting if you're running into issues with it. _Off by default_
  -	**Do face inpaint after hair inpaint:** does what it says. _Off by default_
  -	**Attempt mid-uspcale inpainting with chosen options:** does what it says. Can be helpful for adding an additional level of detail. Off by default
  -	**Use plain Image2Image instead of SD Upscale:** does what it says. _Off by default_
  -	**Don't use SD upscale and inpaint HRfix result. Great for weak GPUs:** besides stated reason to use it, it can be useful to people accustomed to High-Res Fix-only pipeline._Off by default_
- <u>Regulate denoise for each step:</u>
  -	All needed info is already in UI, but i would like to add that rom SD WebUi's version 1.6.0+ necessary denoise for DPM++ samplers is like x2 from DDIM denoise up to 0.5; E.g. 0.2 on DDIM is roughly the same as 0.4 on DPM++ 2M Karras
- <u> Sliders for parameters:</u>
  -	**High-Res Fix scale factor:** all info in UI
  -	**Strength of Filtering:** intensity of Filtering function's effect. 0.3-0.5 works best, higher is tricky, but can be helpful for some artstyles
  -	**Multiplier for LoRA strength lowering:** does what it says. Increase if you want to preserve more of artstyle from your LoRAs
  -	**Face Recognition minimum confidence:** increase for stricter face detection, decrease if having problems on more anime-like artstyles
- <u>**Really advanced options:**</u>
  -	Tile Overlap parameter for SD Upscale, Scaling factor for face inpainting, Scaling factor for eyes inpainting: all info in UI
  - <u>Algorithm's steps' settings:</u> 
    -	**Checkpoint:** allows you to choose different one of your checkpoints to be used on this step. Great for mixing artstyles and combining best qualities of each checkpoint!
    -	**Sampler:** obvious
    -	**Clip Skip:** my use case is to generate base image on Сlip Skip 2 but work with it on later steps on Clip Skip 1 for better realism
    -	**Steps:** obvious
    -	**Prompt & Negative prompt:** allows you to use different prompts and LoRAs for each step. Like, using object or content LoRAs and exclude them from later steps, replacing with LoRAs that have great style, but negatively impact image's content if used in txt2img generation

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/cd6f7365-e4fd-43dc-94c9-fa535c8cc249)



## _Algorithm itself:_
1. Txt2img generation in chosen resolution
2. High-res fix to boost details and gain sharpness
3. [Optional, "on" by default] Filtering with chosen strength 
4. [Optional, "off" by default] Automatic inpainting of face and eyes with chosen parameters
5. SD Upscale 
6. Automatic inpainting of face and eyes with chosen parameters 

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/38c0bed6-84b0-43c5-adf7-14a169b4caf6)
  


## _Coming in 1.0:_
- Release as full extension.
- ControlNet integration.
- More face recognition models (including anime-friendly) 
![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/798a92e9-0105-4b39-85b6-5b89048a108e)

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/2b60ba4f-86af-4c53-a4f3-2d85d3f03e10)

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/4da581ed-3e00-4abc-88e3-f41710f37cee)
