# AutoChar

AutoChar Control Panel is a custom script for Stable Diffusion WebUI by Automatic1111 made to help newbies and enthusiasts alike achieve great pictures with less effort.
Basically, it's automation of my basic SD workflow for illustrations (check 'em here: https://t.me/dreamforge_tg) and also my Bachelor graduation work, for which I got an A. I've put decent emphasis in code readability and comments for it, so I hope it will help future contributors and developers of other scripts. 

So, let us enjoy the open beta!

## Installation
**Just put script and .onnx face recognition model in your stable-diffusion-webui/scripts folder**

## How to use
1. Go to your txt2img tab
2. Write prompt, select basic parameters as usual (you don't highres fix, since it's included in the algorithm)
3. Select "AutoChar Control Panel" in dropdown menu Scripts in the lower part of page
4. Click "Generate" and enjoy

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/798a92e9-0105-4b39-85b6-5b89048a108e)

## _WARNINGS_

-It's kind of conflicting with AfterDetailer and may cause crashes if used together
-It's working poorly with Dynamic Prompts as of now, so it's not advised as well, read Known Issues for details
If it gives you FileNotFoundError, please, make sure your saving directories are divided by txt2img and img2img folders in /stable-diffusion-webui/output. You can do it it Settings-Paths for saving.

## _Functions and features as of now:_
- Automatic 2-stage upscale
- Automatic face and eyes inpaint, for all detected faces on image
- Custom filtering function to help reduce noise on first generation and raise sharpness of the image before upscaling
- For even higher quality, you can toggle face and eyes inpaint in between upscaling steps with little impact on generation time
- If you're really heavy on your LORA's usage, you can toggle CFG scale lowering for face inpaint to avoid burning
- Fully compatible with ControlNet
- Works in both txt2img and img2img (for img2img, it will use selected resolution as base, and Second upscale denoising strength in AutoChar's advanced settings, and ignore one selected in parameters of generation)

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/36794ff7-6c07-4356-8268-28f93ab63556)


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
- Usage of SD upscale rather base img2img for second step
  
![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/4da581ed-3e00-4abc-88e3-f41710f37cee)

## _Known issues, also feel free to suggest your fixes:_
- Poor for using on anime arts, can't detect faces on them
- Sometimes, it finds face on character's breasts or armpits, fix is coming, but not today, so brace yourself for some occasional body horror (but you can see ALL steps in your history, so it's not really a big deal)
- Doesn't pick up your selected Styles from menu automatically, you need to APPLY them to prompt first
- Don't select 4x-UltraSharp upscaler if you don't have one (if so, get it here https://mega.nz/folder/qZRBmaIY#nIG8KyWFcGNTuMX_XNbJ_g)
- BAD with Dynamic Prompts extension, will default to wildcard prompt after first upscale, not sure how to fix this one due to complicated nature of extension. It will apply proper prompt only for txt2img and high-res fix stepts, then for img2img and inpaint it will use FULL prompt, so, depending on variance in wildcards, it can cause issues from same eye colors to horror abominations if there are different fantasy races there
