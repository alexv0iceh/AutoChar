# AutoChar - Version 0.9 is here!
https://civitai.com/models/95923/autochar-easy-character-art-creation-with-face-auto-inpaint

AutoChar Control Panel is a custom script for Stable Diffusion WebUI by Automatic1111 made to help newbies and enthusiasts alike achieve great pictures with less effort.
Basically, it's automation of my basic SD workflow for illustrations (check 'em here: https://t.me/dreamforge_tg) and also my Bachelor graduation work, for which I got an A. I've put decent emphasis in code readability and comments for it, so I hope it will help future contributors and developers of other scripts.  

So, let us enjoy the open beta!

## Installation
**Just put script and .onnx face recognition model in your stable-diffusion-webui/scripts folder. PLEASE, don't try to install via URL, it's not an extension, it won't be visible this way!**

## How to use
1. Go to your txt2img tab
2. Write prompt, select basic parameters as usual (you don't need highres fix, since it's included in the algorithm)
3. Select "AutoChar Beta 0.9" in dropdown menu Scripts in the lower part of page
4. Click "Generate" and enjoy
![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/16919ca6-1de3-4052-a2fd-6729c9e890e5)

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/ce325cd3-9f7d-4c4b-9b79-ffa8df158171)


## _WARNINGS_

- It's kind of conflicting with AfterDetailer and may cause crashes if used together
- If it gives you FileNotFoundError, please, make sure your saving directories are divided by txt2img and img2img folders in /stable-diffusion-webui/output. You can do it it Settings-Paths for saving.

## _Functions and features as of now:_
- Automatic 2-stage upscale
- Automatic face and eyes inpaint, for all detected faces on image
- Custom filtering function to help reduce noise on first generation and raise sharpness of the image before upscaling
- For even higher quality, you can toggle face and eyes inpaint in between upscaling steps with little impact on generation time
- If you're really heavy on your LORA's usage, you can toggle CFG scale lowering for face inpaint to avoid burning
- Fully compatible with ControlNet
- Works in both txt2img and img2img (for img2img, it will use selected resolution as base, and Second upscale denoising strength in AutoChar's advanced settings, and ignore one selected in parameters of generation)
- Added full support for Dynamic Prompts. Enjoy fully automatic enhancement for all of your randomized generations.
- Added SD Upscale as a new default instead of basic Image2Image. Much greater detail and sharpness + customization with your preferred upscaler.
- Lower LoRA: new measure to avoid burnout on faces when using strong or multiple LoRAs. On by default.
- Biggest face: inpaints only the largest face on generation, no more abominable body horrors on armpits and breasts, also helps with crowdy pictures. On by default.
- Now only first and last generation of each cycle will be saved to Txt2img output folder by default. All the other steps will be stored in Img2img folder.

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/d71d32c0-b5fb-4073-83a7-e244fa8f1073)

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/798a92e9-0105-4b39-85b6-5b89048a108e)

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/2b60ba4f-86af-4c53-a4f3-2d85d3f03e10)

## _Algorithm itself:_
- Txt2img generation in chosen resolution
- High-res fix (default: x1.2 on 0.55 denoising strength) to boost details and gain sharpness
- [Optional, "on" by default] Filtering with chosen strength (default: 0.5, less is better for realistic or smooth-rendered images)
- [Optional, "off" by default] Automatic inpainting of face and eyes with chosen parameters (default: 0.2 denoising strength)
- Img2img (default: x1.5 from first generation on 0.3 denoising strength)
- Automatic inpainting of face and eyes with chosen parameters (default: 0.2 denoising strength)

## _Future functions, feel free to wait or suggest your code:_
- Adding anime face recognition model (sadly, it will kinda break or make it less time efficient to upscale eyes)
- More elegant implementation for img2img (it works just fine now, but the inteface is the same when it should be different)
- Add automatic head area hightlighting with added contrast and brightness
- Move information from terminal to textbox in WebUI
- ControlNet fixation of first/second generation to preserve shapes through upscale better
- Selecting flat color for init image, would be great in achieving darker pictures with black or grey init, as well as setting color theme of image
- Integration of postprocessing extras upscale and custom effects, such as noise, vignette or watermarks
  
![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/4da581ed-3e00-4abc-88e3-f41710f37cee)

## _Known issues, also feel free to suggest your fixes:_
- Poor for using on anime arts, can't detect faces on them
- Doesn't pick up your selected Styles from menu automatically, you need to APPLY them to prompt first
- Don't select 4x-UltraSharp upscaler if you don't have one (if so, get it here https://mega.nz/folder/qZRBmaIY#nIG8KyWFcGNTuMX_XNbJ_g)

## _Coming in 1.0:_
- Release as full extension.
- Full Img2Img support.
- ControlNet integration.
- More parameters for advanced users.
