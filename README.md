# AutoChar
Stable Diffusion WebUI extension for fast and easy automatic face and eyes inpainting, featuring many additional benefits and options for creating character art.

AutoChar Control Panel is a custom script for Stable Diffusion WebUI by Automatic1111 made to help newbies and enthusiasts alike achieve great pictures with less effort.
Basically, it's automation of my basic SD workflow for illustrations (check 'em here: https://t.me/dreamforge_tg) and also my Bachelor graduation work, for which I got an A.

INSTALLATION:
Just put script and .onnx face recognition model in your stable-diffusion-webui/scripts folder

![alt text](https://imgur.com/lyhPOEE)

Functions as of now:
- Automatic 2-stage upscale
- Automatic face and eyes inpaint, for all detected faces on image
- Custom filtering function to help reduce noise on first generation and raise sharpness of the image before upscaling
- For even higher quality, you can toggle face and eyes inpaint in between upscaling steps with little impact on generation time
- If you're really heavy on your LORA's usage, you can toggle CFG scale lowering for face inpaint to avoid burning

Future functions, feel free to wait or suggest your code:
- Adding anime face recognition model (sadly, it will kinda break or make it less time efficient to upscale eyes)
- More elegant implementation for img2img (it works just fine now, but generates lowres picture on first step nonetheless)
- ControlNet fixation of first/second generation to preserve shapes through upscale better
- Selecting flat color for init image, would be great in achieving darker pictures with black or grey init, as well as setting color theme of image
- Integration of postprocessing extras upscale and custom effects, such as noise, vignette or watermarks
- Usage of SD upscale rather base img2img for second step

Known issues, also feel free to suggest your fixes:
- Poor for using on anime arts, can't detect faces on them
- Sometimes, it finds face on character's breasts or armpits, fix is coming, but not today, so brace yourself for some occasional body horror (but you can see ALL steps in your history, so it's not really a big deal)
- Don't select 4x-UltraSharp upscaler if you don't have one (if so, get it here https://mega.nz/folder/qZRBmaIY#nIG8KyWFcGNTuMX_XNbJ_g)
- BAD with Dynamic Prompts extension, will default to wildcard prompt after first upscale, not sure how to fix this one due to complicated nature of extension
