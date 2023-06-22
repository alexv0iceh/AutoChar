# AutoChar
Stable Diffusion WebUI extension for fast and easy automatic face and eyes inpainting, featuring many additional benefits and options for creating character art.

AutoChar Control Panel is a custom script for Stable Diffusion WebUI by Automatic1111 made to help newbies and enthusiasts alike achieve great pictures with less effort.
Basically, it's automation of my basic SD workflow for illustrations (check 'em here: https://t.me/dreamforge_tg) and also my Bachelor graduation work, for which I got an A.
So, let us enjoy the open beta!

_**!! INSTALLATION !!:**_
**Just put script and .onnx face recognition model in your stable-diffusion-webui/scripts folder**

_**!!  USAGE  !!:**_
Just go to your **txt2img** tab, write prompt, choose basic parameters (sampler, steps, CFG, etc), choose AutoChar in Scripts falling menu, press Generate and enjoy

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/798a92e9-0105-4b39-85b6-5b89048a108e)

_Functions as of now:_
- Automatic 2-stage upscale
- Automatic face and eyes inpaint, for all detected faces on image
- Custom filtering function to help reduce noise on first generation and raise sharpness of the image before upscaling
- For even higher quality, you can toggle face and eyes inpaint in between upscaling steps with little impact on generation time
- If you're really heavy on your LORA's usage, you can toggle CFG scale lowering for face inpaint to avoid burning

![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/2b60ba4f-86af-4c53-a4f3-2d85d3f03e10)

_Algorithm itself:_
- Txt2img generation in chosen resolution
- High-res fix (default: x1.2 on 0.55 denoising strength) to boost details and gain sharpness
- [Optional, "on" by default] Filtering with chosen strength (default: 0.5, less is better for realistic or smooth-rendered images)
- [Optional, "off" by default] Automatic inpainting of face and eyes with chosen parameters (default: 0.2 denoising strength)
- Img2img (default: x1.5 from first generation on 0.3 denoising strength)
- Automatic inpainting of face and eyes with chosen parameters (default: 0.2 denoising strength)

_Future functions, feel free to wait or suggest your code:_
- Adding anime face recognition model (sadly, it will kinda break or make it less time efficient to upscale eyes)
- More elegant implementation for img2img (it works just fine now, but generates lowres picture on first step nonetheless)
- Move information from terminal to textbox in WebUI
- ControlNet fixation of first/second generation to preserve shapes through upscale better
- Selecting flat color for init image, would be great in achieving darker pictures with black or grey init, as well as setting color theme of image
- Integration of postprocessing extras upscale and custom effects, such as noise, vignette or watermarks
- Usage of SD upscale rather base img2img for second step
- 
![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/dca5b859-f464-4e20-b9b9-a6ed9516bd48)


![image](https://github.com/alexv0iceh/AutoChar/assets/74978526/4da581ed-3e00-4abc-88e3-f41710f37cee)

_Known issues, also feel free to suggest your fixes:_
- Poor for using on anime arts, can't detect faces on them
- Sometimes, it finds face on character's breasts or armpits, fix is coming, but not today, so brace yourself for some occasional body horror (but you can see ALL steps in your history, so it's not really a big deal)
- Don't select 4x-UltraSharp upscaler if you don't have one (if so, get it here https://mega.nz/folder/qZRBmaIY#nIG8KyWFcGNTuMX_XNbJ_g)
- BAD with Dynamic Prompts extension, will default to wildcard prompt after first upscale, not sure how to fix this one due to complicated nature of extension
