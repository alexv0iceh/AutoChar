import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from modules.shared_state import State
from modules.ui_components import FormRow
from modules.shared_cmd_options import cmd_opts
from modules.shared_options import options_templates
from modules import scripts_postprocessing, shared
from modules.processing import StableDiffusionProcessingImg2Img
from modules.ui_common import create_refresh_button
from modules import processing, shared, sd_samplers, images, devices,shared_items
from modules import sd_models, sd_vae
from modules import styles

# Check OpenCV version and update if necessary
import cv2
import subprocess
import pkg_resources

def update_opencv():
    subprocess.check_call(["python", '-m', 'pip', 'install', '--upgrade', 'opencv-python'])

# Get the current OpenCV version
current_version = cv2.__version__
latest_version = pkg_resources.get_distribution("opencv-python").version

# If the current version is not the latest, update OpenCV
if current_version != latest_version:
    print(f"Updating OpenCV from version {current_version} to {latest_version}")
    update_opencv()
    print("Update complete.")


#print(sd_models.checkpoints_list)
#print(sd_vae.vae_dict)

class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "AutoChar 0.9.5"

    def ui(self, is_img2img):
        gr.Markdown(
                """
                ## Welcome to AutoChar!
                ### If you're new to it, feel free to take a look at the official guide for this version: https://www.youtube.com/watch?v=jNUMHtH1U6E
                """)

        scale_factor = gr.Slider(minimum=1.0, maximum=5, step=0.1, value=2,
                                 info= "Desired output image scale relative to resolution chosen in main interface",
                                 label="Final scale factor")
        
        with FormRow(variant="compact",equal_height=True):
            options = gr.CheckboxGroup(label="Options", choices=['Automatic face inpaint', 'Automatic eyes inpaint',
                                                                'Display info in terminal'],
                                    value=['Automatic face inpaint', 'Automatic eyes inpaint'], 
                                    info= "Choose options for AutoChar algorithm",
                                    elem_id=self.elem_id("options"))
            mode = gr.Radio(label="Operating mode", choices=['Txt2Img', 'Img2Img'], 
                                    value="Txt2Img", 
                                    info= "Choose operating mode. Txt2Img will use full pipeline, Img2Img will skip first steps before SD Upscale",
                                    elem_id=self.elem_id("mode"))

        with FormRow(variant="compact",equal_height=True):
            ui_upscaler_1 = gr.Dropdown(
                choices=[*shared.latent_upscale_modes, *[x.name for x in shared.sd_upscalers]],
                label="Upscaler 1 (High-Res Fix)",
                value="Latent (bicubic antialiased)",
                info= "Latents are great for adding details, for consistency with basic generation use any -GAN or 4x-UltraSharp",
                elem_id=self.elem_id("ui_upscaler_1")
            )


            ui_upscaler_2 = gr.Dropdown(
                [x.name for x in shared.sd_upscalers], label="Upscaler 2 (SD Upscale)",
                value="R-ESRGAN 4x+",
                info= "Highly recommended to download and use 4x-UltraSharp for general purposes",
                type="index",
                elem_id=self.elem_id("ui_upscaler_2")
            )


        with gr.Accordion('Advanced options', open=False):
            gr.Markdown(
                """### <u> Algorithm additional functions </u>  """)
            with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """ ##### Quality functions """)
                        filtering = gr.Checkbox(True, label="Filtering function")

                        biggest_face = gr.Checkbox(True, label="Inpaint only the biggest face on the image")

                        lower_lora_param = gr.Checkbox(True,
                                                label="Lower LoRA strength for face inpaint. Helps avoid burnout with strong LORAs")
                        use_ddim = gr.Checkbox(True, label="Use DDIM sampler for better inpaint. Will use chosen in interface otherwise")

                        lower_cfg = gr.Checkbox(False, label="Lower CFG for face inpaint. Helps avoid burning with multiple LoRAs")

                    with gr.Column():
                        gr.Markdown(
                            """ ##### Algorithm-alterting functions """)
                        inpaint_hair = gr.Checkbox(False, label="Make face inpaint box larger to inpaint hair along with the face")

                        inpaint_hair_and_face = gr.Checkbox(False, label="Do face inpaint after hair inpaint")

                        mid_inpainting = gr.Checkbox(False, label="Attempt mid-uspcale inpainting with chosen options")

                        use_img2img = gr.Checkbox(False, label="Use plain Image2Image instead of SD Upscale")

                        do_not_sd_upscale = gr.Checkbox(False, label="Don't use SD upscale and inpaint HRfix result. Great for weak GPUs")
            gr.Markdown(
                """
                #### <u> Regulate denoise for each step </u>
                """)
            with FormRow(variant="compact"):
                first_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                                                label="High-Res Fix denoising strength", info="0.45-0.6 for Latents, 0.2-0.4 for all others")
                face_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                            label="Face inpainting denoising strength", info="Higher for smaller faces, lower for bigger. Also, lower for anime styles")

            with FormRow(variant="compact"):        
                second_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                                                label="SD Upscale denoising strength")
                eyes_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                                            label="Eyes inpainting denoising strength")
                    
            gr.Markdown(
                """
                #### <u> Sliders for parameters </u>
                """)
            
            scale_factor0 = gr.Slider(minimum=1.0, maximum=3, step=0.05, value=1.25,
                    label="High-Res Fix scale factor", info="For bigger scales Latents will go wild and produce artifacts and new limbs, consider it before increasing")

            with FormRow(variant="compact"):

                    strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                                                label="Strength of Filtering")
                    lora_lowering = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.35,
                                                label="Multiplier for LoRA strength lowering")

                    face_confidence_threshold = gr.Slider(label="Face Recognition minimum confidence", minimum=0.0, maximum=1.0, step=0.05, value=0.7)

            with gr.Accordion('REALLY advanced options (Brain Damage alert)', open=False):
                with FormRow(variant="compact"):
                    overlap = gr.Slider(label="Tile Overlap parameter for SD Upscale", minimum=0, maximum=256, step=16, value=64, elem_id=self.elem_id("overlap"), info="Can be helpful to regulate time and VRAM consumption on SD upscale step")
                    face_resolution_scale_slider = gr.Slider(minimum=1.0, maximum=4, step=0.1, value=2.5,
                        label="Scaling factor for face inpainting", info="Higher = sharper and crispier face, but higher time and VRAM consumption")   
                    eyes_resolution_scale_slider = gr.Slider(minimum=1.0, maximum=3, step=0.1, value=1.5,label="Scaling factor for eyes inpainting",
                        info="Note that it uses enlarged face as base, so the modifer is lower")   
                gr.Markdown(
                """
                #### <u> High-Res Fix settings </u>
                """)
                with FormRow(variant="compact"):
                    hr_model = gr.Dropdown( choices=["Use same checkpoint"] + sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint", label='Checkpoint for High-Res Fix')
                    create_refresh_button(hr_model, sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + sd_models.checkpoint_tiles(use_short=True)}, "hr_checkpoint_refresh")

                    hr_sampler = gr.Dropdown(
                        [x.name for x in shared_items.list_samplers()], label="High-Res Fix sampler",
                        value="DPM++ 2M Karras",
                        elem_id=self.elem_id("hr_sampler")
                    )
                    first_steps = gr.Slider(minimum=0, maximum=100, step=1, value=12,
                                                label="High-Res Fix steps")

                    #hr_upscale_vae = gr.Dropdown(choices=['None', *sd_vae.vae_dict.keys()], label='VAE for High-Res Fix', value='None')    
                with FormRow(variant="compact"):
                    highres_prompt = gr.Textbox(label="High-Res Fix prompt", lines=3, placeholder="Prompt for High-Res Fix pass.\nLeave empty to use the same prompt as in main textbox.",scale=2)
                    highres_negprompt = gr.Textbox(label="High-Res Fix negative prompt", lines=3, placeholder="Negative Prompt for High-Res Fix pass.\nLeave empty to use the same prompt as in main textbox." )

                gr.Markdown(
                """
                #### <u> SD Upscale settings </u>
                """)
                with FormRow(variant="compact"):
                    sd_upscale_model = gr.Dropdown( choices=["Use same checkpoint"] + sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint", label='Checkpoint for SD Upscale', scale=2)
                    create_refresh_button(sd_upscale_model, sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + sd_models.checkpoint_tiles(use_short=True)}, "sd_upscale_checkpoint_refresh")

                    sd_upscale_sampler = gr.Dropdown(
                        [x.name  for x in shared_items.list_samplers()], label="SD Upscale sampler",
                        value="DPM++ 2M Karras",
                        elem_id=self.elem_id("sd_upscale_sampler"), scale=2
                    )
                    second_clip = gr.Slider(minimum=1, maximum=12, step=1, value=1,
                                                label="Clip Skip")
                    second_steps = gr.Slider(minimum=0, maximum=100, step=1, value=12, scale=2,
                                                label="SD Upscale steps")

                    #sd_upscale_vae = gr.Dropdown(choices=['None', *sd_vae.vae_dict.keys()], label='VAE for upscale', value='None')     
                with FormRow(variant="compact"):
                    sd_upscale_prompt = gr.Textbox(label="SD Upscale prompt", lines=3, placeholder="Prompt for SD Upscale pass.\nLeave empty to use the same prompt as in main textbox.", scale=2)
                    sd_upscale_negprompt = gr.Textbox(label="SD Upscale negative prompt", lines=3, placeholder="Negative Prompt for SD Upscale pass.\nLeave empty to use the same prompt as in main textbox.")

                gr.Markdown(
                """
                #### <u> Inpaint settings </u>
                """)
                with FormRow(variant="compact"):
                    face_model = gr.Dropdown( choices=["Use same checkpoint"] + sd_models.checkpoint_tiles(use_short=True), value="Use same checkpoint", label='Checkpoint for inpaint', scale=2)
                    create_refresh_button(face_model, sd_models.list_models, lambda: {"choices": ["Use same checkpoint"] + sd_models.checkpoint_tiles(use_short=True)}, "face_checkpoint_refresh")
                    
                    face_sampler = gr.Dropdown(
                        [x.name for x in shared_items.list_samplers()], label="Inpaint sampler",
                        value="DPM++ 2M Karras",
                        elem_id=self.elem_id("face_sampler"), scale=2
                    )
                    face_clip = gr.Slider(minimum=1, maximum=12, step=1, value=1,
                                                label="Clip Skip")
                    face_steps = gr.Slider(minimum=0, maximum=100, step=1, value=12,
                            label="Inpaint steps", scale=2)
                    
                with FormRow(variant="compact"):
                    inpaint_prompt = gr.Textbox(label="Inpaint prompt", lines=3, placeholder="Prompt for Inpaint passes.\nLeave empty to use the same prompt as in main textbox.",scale=2 )
                    inpaint_negprompt = gr.Textbox(label="Inpaint negative prompt", lines=3, placeholder="Negative Prompt for Inpaint passes.\nLeave empty to use the same prompt as in main textbox.")
                    #face_vae = gr.Dropdown(choices=['None', *sd_vae.vae_dict.keys()], label='VAE for inpaint', value='None') 
        gr.Markdown(
        """
        ### In case of questions or troubles, visit https://civitai.com/models/95923
        """)    

        return [filtering, strength, ui_upscaler_1, first_denoise, second_denoise, face_denoise, eyes_denoise, options,mode,
                mid_inpainting, scale_factor, scale_factor0, lower_cfg,lower_lora_param,biggest_face,ui_upscaler_2,overlap,use_img2img,lora_lowering,
                face_confidence_threshold,inpaint_hair, inpaint_hair_and_face,use_ddim, do_not_sd_upscale,
                inpaint_prompt,inpaint_negprompt,highres_prompt,highres_negprompt, sd_upscale_prompt,sd_upscale_negprompt,first_steps, hr_sampler, second_steps,sd_upscale_sampler,face_steps, face_sampler,
                hr_model,sd_upscale_model,face_model,face_resolution_scale_slider,eyes_resolution_scale_slider, second_clip,face_clip]

    def run(self, p, filtering, strength, ui_upscaler_1, first_denoise, second_denoise, face_denoise, eyes_denoise,options, mode,
            mid_inpainting, scale_factor, scale_factor0, lower_cfg,lower_lora_param,biggest_face,ui_upscaler_2,overlap,use_img2img,lora_lowering,face_confidence_threshold,
            inpaint_hair,inpaint_hair_and_face,use_ddim, do_not_sd_upscale,inpaint_prompt,inpaint_negprompt,highres_prompt,highres_negprompt, sd_upscale_prompt,sd_upscale_negprompt,
            first_steps, hr_sampler, second_steps,sd_upscale_sampler,face_steps, face_sampler,
            hr_model,sd_upscale_model,face_model,face_resolution_scale_slider,eyes_resolution_scale_slider, second_clip,face_clip):
        
        
        initial_seed_and_info = [None, None, None]
        face_inpaint_flag = True if "Automatic face inpaint" in options else False
        eyes_inpaint_flag = True if "Automatic eyes inpaint" in options else False
        info_flag = True if "Display info in terminal" in options else False
        if "Img2Img" in mode:
            filtering = False
        i2i_only_flag = True if "Img2Img" in mode else False
        mid_face_flag = None
        mid_eyes_flag = None
        inpaint_hair_flag = None
        inpaint_hair_and_face_flag = None
        use_ddim_flag = None

        if info_flag:
            print(options)
            print('filtering', filtering, 'strength', strength, 'ui_upscaler_1', ui_upscaler_1, 'face_inpaint_flag',
                  face_inpaint_flag, 'eyes_inpaint_flag', eyes_inpaint_flag, 'scale_factor', scale_factor)

        if mid_inpainting:
            mid_face_flag = face_inpaint_flag
            mid_eyes_flag = eyes_inpaint_flag

        if inpaint_hair:
            inpaint_hair_flag = True

        if inpaint_hair_and_face:
            inpaint_hair_and_face_flag = True

        if use_ddim:
            use_ddim_flag = True


        upscaler = ui_upscaler_1

        from PIL import Image
        
        import cv2
        import numpy as np
        import torch
        import math
        import re
        import random

        initial_prompt = p.prompt
        initial_seed = None
        initial_info = None

        is_last = False

        all_images = []
        pos = 0
        batch_count = p.n_iter
        p.n_iter = 1
        State.job_count = batch_count

        instance_img2img = StableDiffusionProcessingImg2Img()
        instance_img2img.outpath_samples = opts.outdir_img2img_samples
        instance_inpaint = StableDiffusionProcessingImg2Img()
        instance_inpaint.outpath_samples = opts.outdir_img2img_samples
        instance_sd_upscale = StableDiffusionProcessingImg2Img()
        instance_sd_upscale.outpath_samples = opts.outdir_img2img_samples

        #styles.StyleDatabase.apply_styles_to_prompt(p.prompt, styles)

        # Function for rounding resolution
        def closest(value, divider):
            return min((i for i in range(value - divider + 1, value + divider - 1)
                        if i % divider == 0),
                       key=lambda x: abs(x - value))

        
        # Function for scaling small eye boxes
        def proportional_scaling(width, height, factor, threshold):

            step = 8  # 8 or 64
            # Calculate the aspect ratio of the image
            aspect_ratio = width / height

            # Scale the width and height by the given factor
            width = width * factor
            height = height * factor

            # Ensure that the resulting width and height are within the given threshold
            if width > threshold:
                # print(f" Proportional Scaling hit the limit: width {width} is larger than {threshold}")
                width = threshold
                height = width / aspect_ratio

            if height > threshold:
                # print(f" Proportional Scaling hit the limit: height {height} is larger than {threshold}")
                height = threshold
                width = height * aspect_ratio

            width = int(width)
            height = int(height)

            # Output height and width suitable for generation
            height = int(math.ceil(float(height) / float(step))) * step
            width = int(math.ceil(float(width) / float(step))) * step

            return (width, height)

        # Function for filtering of images
        def enhance_image(image, strength):
            import numpy as np
            import cv2
            # Parameters:
            # image_path - path to image
            # strength - desired strength of filter, from 0 to 1,  default = 1
            # np_frame = np.array(image.images[0].convert("RGB"))
            image = np.array(image.images[0])
            # Detail enhance
            dst = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15)
            # sigma_s controls how much the image is smoothed - the larger its value,
            # the more smoothed the image gets, but it's also slower to compute.
            # sigma_r is important if you want to preserve edges while smoothing the image.
            # Small sigma_r results in only very similar colors to be averaged (i.e. smoothed), while colors that differ much will stay intact.
            # Sharpening kernel init
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]])
            # Sharpening
            dst2 = cv2.filter2D(image, -1, kernel_sharpening)
            # Blending detailed and sharpened images
            blended = cv2.addWeighted(dst, 0.6, dst2, 0.4, 0)
            # Denoising
            denoised = cv2.fastNlMeansDenoisingColored(blended, None, 5, 5, 7, 14)
            # Blending with the original
            denoised_blended = cv2.addWeighted(image, 1 - strength, denoised, strength, 0)
            if info_flag:
                print('Filtering complete')
            return Image.fromarray(denoised_blended)

        # Function for face detection and mask creation
        def mask_create(image,hair_inpaint_flag_param):

            #  Regulating parameters
            face_resolution_scale = face_resolution_scale_slider
            resolution_scale = eyes_resolution_scale_slider
            if hair_inpaint_flag_param:
                face_resolution_scale = face_resolution_scale_slider*1.5
            mask_dilation = 1.5
            facebox_size_multiplier = 1.25
            face_found = True
            rotate = False
            directory = os.path.dirname(__file__)
            image = np.array(image)

            # Load the model

            weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
                
            face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0), face_confidence_threshold)

            # Face detection

            channels = 1 if len(image.shape) == 2 else image.shape[2]
            if channels == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if channels == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            height, width, _ = image.shape
            face_detector.setInputSize((width, height))
            face_height_multiplier = 1

            _, faces = face_detector.detect(image)
            faces = faces if faces is not None else []

            faces_quantity = len (faces)
            # Checking if the face was found
            if faces_quantity == 0:
                print("Face detection failed! Try more realistic picture.")
                face_found = False


            if biggest_face and faces_quantity > 1:
                # Find the biggest face in the array of faces
                # Size is determined by face[2] - width and face[3] - height of face box
                biggest_face_size = 0
                biggest_face_index = 0
                for i, face in enumerate(faces):
                    face_size = face[2] * face[3]
                    if face_size > biggest_face_size:
                        biggest_face_size = face_size
                        biggest_face_index = i
                # Keep only the biggest face
                faces = [faces[biggest_face_index]]
                faces_quantity = 1


            # Checking if the face box is horizontal
            if faces_quantity == 1:
                face1 = faces[0]
                if face1[2] > face1[3]:
                    face_height_multiplier = 1.2
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    rotate = True
                    height, width, _ = image.shape
                    face_detector.setInputSize((width, height))
                    _, horizontalFaces = face_detector.detect(image)
                    if horizontalFaces is not None:
                        faces = horizontalFaces
                        print(faces)

            results = []
            if faces is not None:
                for face in faces:
                    face_width = face[2]


                    # finding higher eye to measure eye y-coordinate difference
                    if face[5] > face[7]:
                        higher_eye = (face[4], face[5])
                        lower_eye = (face[6], face[7])
                    else:
                        higher_eye = (face[6], face[7])
                        lower_eye = (face[4], face[5])
                    eye_y_dif = abs(int(higher_eye[1] / lower_eye[1]))
                    eye_x_distance = abs(int(higher_eye[0] - lower_eye[0]))
                    # if eye_x_distance >= 0.4 * face_width:
                    if abs(face[4] - face[8]) >= 0.15 * face_width and abs(face[6] - face[8]) >= 0.15 * face_width:
                        # front or 3/4 view
                        eye_box_corner = (
                            int(face[4] - eye_x_distance * 0.55),
                            int(higher_eye[1] - eye_x_distance * 0.25 * (1.5 * eye_y_dif)))
                        face_height_multiplier = face_height_multiplier * 1.2
                        eye_y_dif_multiplier = 1.25
                    else:
                        # profile view
                        eye_box_corner = (
                            int(face[4] - eye_x_distance),
                            int(higher_eye[1] - eye_x_distance * 0.45 * (1.4 * eye_y_dif)))
                        face_height_multiplier = face_height_multiplier * 1.3
                        eye_y_dif_multiplier = 1.5
                        if (face[4]) < (face[0] + 0.5 * face_width):  # left eye profile view
                            eye_box_corner = (
                                int(face[4] - eye_x_distance * 0.1),
                                int(higher_eye[1] - eye_x_distance * 0.45 * (1.65 * eye_y_dif)))
                        eye_x_distance = eye_x_distance * 1.2
                    box = list(map(int, face[:4]))

                    eye_box = [eye_box_corner[0], eye_box_corner[1], int(eye_x_distance * 2),
                            int((eye_x_distance * 0.8) * (eye_y_dif_multiplier * eye_y_dif))]
                    color = (0, 0, 255)

                    thickness = 2
                    cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
                    cv2.rectangle(image, eye_box, (0, 255, 255), thickness, cv2.LINE_AA)
                    cv2.circle(image, eye_box_corner, 5, (0, 255, 255), -1, cv2.LINE_AA)

                    if rotate:
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    mask1 = np.zeros(image.shape[:2], dtype="uint8")
                    mask2 = np.zeros(image.shape[:2], dtype="uint8")
                    mask3 = np.zeros(image.shape[:2], dtype="uint8")

                    hair_box = box.copy()
                    hair_box[0] -= int(box[0]*0.5)
                    hair_box[1] -= int(box[1]*0.5)

                    box = [box[0], box[1], closest(int(box[2] * 1.25), 4),
                        closest(int(box[3] * face_height_multiplier), 4)]
                
                    hair_box = [hair_box[0], hair_box[1], closest(int(hair_box[2] * 4.25), 4),
                        closest(int(hair_box[3] * face_height_multiplier *1.5), 4)]
                    
                    cv2.rectangle(mask1, box, 255, -1)
                    cv2.rectangle(mask2, eye_box, 255, -1)
                    cv2.rectangle(mask3, hair_box, 255, -1)

                    inpaint_face_size = [closest(int(box[2] * face_resolution_scale), 4),
                                        closest(int(box[3] * face_resolution_scale), 4)]
                    inpaint_eye_size = list(
                        [closest(int(eye_box[2] * resolution_scale), 4), closest(int(eye_box[3] * resolution_scale), 4)])
                    
                    if info_flag and (inpaint_face_size[0] < 384 or inpaint_face_size[1] < 384):
                            print('Face box too small! Initiating proportional_scaling')
                    while inpaint_face_size[0] < 384 or inpaint_face_size[1] < 384:
                        inpaint_face_size = list(
                            proportional_scaling(inpaint_face_size[0], inpaint_face_size[1], 1.1, 2048))
                    if info_flag and (inpaint_eye_size[0] < 384 or inpaint_eye_size[1] < 384):
                            print('Eye box too small! Initiating proportional_scaling')
                    while inpaint_eye_size[0] < 384 or inpaint_eye_size[1] < 384:
                        inpaint_eye_size = list(proportional_scaling(inpaint_eye_size[0], inpaint_eye_size[1], 1.2, 2048))
                    if info_flag:
                        print("Resolution for face inpaint:\n" + str(inpaint_face_size), 'type:', type(inpaint_face_size))
                        print("Resolution for eye inpaint:\n" + str(inpaint_eye_size), 'type:', type(inpaint_eye_size))
                    results.append(
                        (Image.fromarray(mask1), Image.fromarray(mask2),Image.fromarray(mask3), inpaint_face_size, inpaint_eye_size, face_found, faces_quantity))


            return results

        # Function for lowering LORA strength
        def lower_lora(current_prompt):
            loras = re.findall(r'(<lora:[^>]*(:(\d*\.?\d*)>))', current_prompt)# find all LoRAs and save them into capture groups. [0] full LoRA = "<lora:name:0.5>", [1] suffix only = ":0.5>", [2] strength only = "0.5"
            new_prompt = current_prompt
            if info_flag:
                print("found LoRAs:", loras)
            for lora in loras:
                lora_strength = round(float(lora[2]) * lora_lowering,3)
                lora_suffix = ':'+str(lora_strength)+'>' # extra step to prevent potentially breaking LoRAs that have numbers in them
                new_lora = lora[0].replace(lora[1], lora_suffix)
                new_prompt = new_prompt.replace(lora[0], new_lora)
            return new_prompt

        # Custom wrapper for process_images(p) to start txt2img+hrfix
        def text2img_hr(upscaler, scale_factor0):

            # print the input
            if info_flag:
                print('txt2img+hr fix \n upscaler:', upscaler)
 
            # Change parameters
            p.enable_hr = True
            p.denoising_strength = first_denoise
            p.hr_scale = scale_factor0
            p.hr_upscaler = upscaler
            p.hr_second_pass_steps = first_steps
            p.sampler_name = hr_sampler
            if highres_prompt:
                p.hr_prompt = highres_prompt
            if highres_negprompt:
                p.hr_negative_prompt = highres_negprompt
            if hr_model == 'Use same checkpoint':
                p.hr_checkpoint_name = None
            else: p.hr_checkpoint_name = hr_model

            if i2i_only_flag:
                p.steps = 1
                p.enable_hr = False
            # Start generation with high-res fix
            hr_output = process_images(p)

            # Write down the seed info for reproducibility
            initial_seed_and_info[0] = hr_output.seed
            initial_seed_and_info[1] = hr_output.info
            initial_seed_and_info[2] = hr_output.prompt
            

            # Print confirmation
            if info_flag:
                print('HR fix complete')

            # Clear cache
            torch.cuda.empty_cache()

            return hr_output

        # Custom wrapper for process_images(p) to start img2img
        def img2img(init_image, scale_factor, is_last_img2img):

            # Print the input
            if info_flag:
                print(f"Input size for img2img: {init_image.images[0].size}")

            if sd_upscale_model != 'Use same checkpoint':
                instance_img2img.refiner_checkpoint = sd_upscale_model
                instance_img2img.refiner_switch_at = 0.01
            # Clear cache
            torch.cuda.empty_cache()

            # Check if it's our last step
            if is_last_img2img:
                instance_img2img.outpath_samples = opts.outdir_txt2img_samples

            # Check for changes in prompt (DYNAMIC PROMPTS PROBLEM)
            #print(instance_img2img.prompt)
            if prompt_temp != initial_prompt:

                print('Prompt is different, assigning Wildcard result')

                instance_img2img.prompt = prompt_temp
                #print(type(instance_img2img.prompt))

            else: print ('Prompt is the same')
            #print(instance_img2img.prompt)
            # Change parameters

            #print(instance_img2img.prompt)

            instance_img2img.negative_prompt = init_image.negative_prompt
            if sd_upscale_prompt:
                instance_img2img.prompt = sd_upscale_prompt
            if sd_upscale_negprompt:
                instance_img2img.negative_prompt = sd_upscale_negprompt
            instance_img2img.seed = init_image.seed
            instance_img2img.init_images = [init_image.images[0]]
            instance_sd_upscale.clip_skip = second_clip
            instance_img2img.denoising_strength = second_denoise
            instance_img2img.steps = second_steps

            # Change resolution so that it's surely dividable by 4
            instance_img2img.width = closest(int(scale_factor * p.width), 4)
            instance_img2img.height = closest(int(scale_factor * p.height), 4)

            # Print new resolution
            if info_flag:
                print('img2img resolution: ', instance_img2img.width, instance_img2img.height)

            # Run img2img
            img2img_output = process_images(instance_img2img)

            # Print confirmation
            if info_flag:
                print('img2mg finished!')

            # Clear cache
            torch.cuda.empty_cache()

            # Reset saving path
            if is_last_img2img:
                instance_img2img.outpath_samples = opts.outdir_img2img_samples

            return img2img_output

        # Custom wrapper for SD uspcale
        def sd_upscale(init_image, scale_factor,is_last_sd_upscale,overlap_func, ui_upscaler_2_func):
            # Print the input
            if info_flag:
                print(f"Input size for sd_upscale: {init_image.images[0].size}")

            from scripts.sd_upscale import Script as sd_up_run

            if sd_upscale_model != 'Use same checkpoint':
                instance_sd_upscale.refiner_checkpoint = sd_upscale_model
                instance_sd_upscale.refiner_switch_at = 0.01

            # Clear cache
            torch.cuda.empty_cache()

            # Check if it's our last step
            if is_last_sd_upscale:
                instance_sd_upscale.outpath_samples = opts.outdir_txt2img_samples


            # Check for changes in prompt (DYNAMIC PROMPTS PROBLEM)
            if prompt_temp != initial_prompt:
                instance_sd_upscale.prompt = prompt_temp
            # Change parameters
            instance_sd_upscale.negative_prompt = init_image.negative_prompt
            if sd_upscale_prompt:
                instance_sd_upscale.prompt = sd_upscale_prompt
            if sd_upscale_negprompt:
                instance_sd_upscale.negative_prompt = sd_upscale_negprompt

            instance_sd_upscale.seed = init_image.seed
            instance_sd_upscale.init_images = [init_image.images[0]]
            instance_sd_upscale.extra_generation_params["SD upscale overlap"] = 64
            instance_sd_upscale.extra_generation_params["SD upscale upscaler"] = shared.sd_upscalers[ui_upscaler_2_func].name
            instance_sd_upscale.clip_skip = second_clip
            instance_sd_upscale.steps = second_steps
            instance_sd_upscale.denoising_strength = second_denoise
            instance_sd_upscale.do_not_save_grid = True
            instance_sd_upscale.do_not_save_samples = True
            instance_sd_upscale.sampler_name = sd_upscale_sampler


            # Change resolution so that it's surely dividable by 4
            #instance_sd_upscale.width = closest(int(scale_factor * p.width), 4)
            #instance_sd_upscale.height = closest(int(scale_factor * p.height), 4)
            #instance_sd_upscale.width, instance_sd_upscale.height = closest_ratio_preserving(int(scale_factor * p.width),int(scale_factor * p.height), 4)

            sd_upscale_scale_factor = round(scale_factor/ scale_factor0, 3)

            # Print new resolution
            if info_flag:
                print('SD upscale resolution: ', sd_upscale_scale_factor)


            # Run sd_upscale
            #sd_upscale_output = sd_up_run.run(instance_sd_upscale)
            sd_upscale_output = sd_up_run.run(self, instance_sd_upscale, None , overlap_func, ui_upscaler_2_func, sd_upscale_scale_factor)


            # Print confirmation
            if info_flag:
                print('SD upscale finished!')

            # Clear cache
            torch.cuda.empty_cache()

            # Reset saving path
            if is_last_sd_upscale:
                instance_sd_upscale.outpath_samples = opts.outdir_img2img_samples

            return sd_upscale_output
        
        # Custom wrapper for process_images(p) to start inpaint
        def inpaint(init_image, p_mask, w, h, denoise,is_last_inpaint, rewrite_seed=False):


            if face_model != 'Use same checkpoint':
                instance_inpaint.refiner_checkpoint = face_model
                instance_inpaint.refiner_switch_at = 0.01

            print("Inpaint debug flag")
            # Change parameters
            if not rewrite_seed:
                instance_inpaint.seed = init_image.seed
            else:
                instance_inpaint.seed = init_image.seed + 10000


            
            # Check for changes in prompt (DYNAMIC PROMPTS PROBLEM)
            if prompt_temp != initial_prompt:
                instance_inpaint.prompt = prompt_temp

            
            instance_inpaint.negative_prompt = init_image.negative_prompt

            if inpaint_prompt:
                instance_inpaint.prompt = inpaint_prompt
            if inpaint_negprompt:
                instance_inpaint.negative_prompt = inpaint_negprompt

            if lower_cfg:
                if instance_inpaint.cfg_scale != p.cfg_scale:
                    instance_inpaint.cfg_scale = p.cfg_scale
                instance_inpaint.cfg_scale = int(instance_inpaint.cfg_scale * 0.66)
                if info_flag:
                    print('Lowering CFG for inpaint \n new CFG:', instance_inpaint.cfg_scale)

            if lower_lora_param:
                new_prompt = lower_lora(instance_inpaint.prompt)
                instance_inpaint.prompt = new_prompt
                #print(instance_inpaint.prompt)
                if info_flag:
                    print('Lowering LORA strength for inpaint \n new LORA strengths:', new_prompt)

            # Check if it's our last step

            if is_last_inpaint:
                instance_inpaint.outpath_samples = opts.outdir_txt2img_samples
            if i2i_only_flag:
                instance_inpaint.outpath_samples = opts.outdir_img2img_samples

            instance_inpaint.clip_skip = face_clip
            instance_inpaint.seed = init_image.seed
            instance_inpaint.init_images = [init_image.images[0]]
            instance_inpaint.image_mask = p_mask
            instance_inpaint.mask_blur = int(w * 0.01)
            instance_inpaint.inpainting_fill = 1
            instance_inpaint.inpaint_full_res = True
            instance_inpaint.inpaint_full_res_padding = int(w * 0.04)
            instance_inpaint.denoising_strength = denoise
            instance_inpaint.sampler_name = face_sampler
            instance_inpaint.steps = face_steps
            instance_inpaint.width = w
            instance_inpaint.height = h

            if use_ddim_flag:
                instance_inpaint.sampler_name = 'DDIM'
                if instance_inpaint.steps<20:
                    instance_inpaint.steps = 20


            # Print inpaint parameters
            if info_flag:
                print('Inpaint parameters: ', instance_inpaint.sampler_name, instance_inpaint.steps, instance_inpaint.width, instance_inpaint.height)

            # Run inpaint
            inpaint_output = process_images(instance_inpaint)


            # Clear cache
            torch.cuda.empty_cache()

            # Reset saving path

            if is_last_inpaint:
                instance_inpaint.outpath_samples = opts.outdir_img2img_samples

            return inpaint_output

        # the job itself
        for n in range(batch_count):
            # Reset to original init image at the start of each batch
            State.job = f"batch {n} of {batch_count} batches \n"
            #State.sampling_steps = p.steps + second_steps + (face_steps*2)
            print(State.job)
            print(p.sampler_name)
            last_image_batch = text2img_hr(upscaler, scale_factor0)
            #hf_fix_output_str = ' '.join([str(elem) for i, elem in enumerate(hr_fix_output.info)])
            #prompt_temp = hr_fix_output.info
           # prompt_temp = hr_fix_output.info.split(('egative prompt:')[0])
            prompt_temp = re.split('Negative prompt: ',last_image_batch.info)[0]
            #print(prompt_temp)
            #print (initial_prompt)
            if i2i_only_flag:
                last_image_batch.images = p.init_images
            if filtering:
                last_image_batch.images[0] = enhance_image(last_image_batch, strength)



            if mid_face_flag:
                for (mask_face, mask_eyes, mask_hair, inpaint_face_size, inpaint_eye_size, face_found, faces_quantity) in mask_create(
                        last_image_batch.images[0],inpaint_hair_flag):
                    if face_found:
                        if info_flag:
                            print('Mid-uspcale face inpaint started: ')

                        mid_image_face_inpaint = inpaint(last_image_batch, mask_face, inpaint_face_size[0],
                                                         inpaint_face_size[1],
                                                         face_denoise,is_last, True)

                        if mid_eyes_flag:
                            if info_flag:
                                print('Mid-uspcale eyes inpaint started: ')
                            mid_image_eyes_inpaint = inpaint(mid_image_face_inpaint, mask_eyes, inpaint_eye_size[0],
                                                             inpaint_eye_size[1], eyes_denoise,is_last, True)
                            last_image_batch.images[0] = mid_image_eyes_inpaint.images[0]
                        else:
                            last_image_batch.images[0] = mid_image_face_inpaint.images[0]


            if not face_inpaint_flag:
                is_last = True

            if face_inpaint_flag:
                is_last = True
                for (mask_face, mask_eyes, mask_hair, inpaint_face_size, inpaint_eye_size, face_found, faces_quantity) in mask_create(
                            last_image_batch.images[0],inpaint_hair_flag):
                    if face_found:
                        is_last = False
                        break
            
            if info_flag:
                if is_last:
                    print('NO Face found in pre upscale check. Marking Upscale as last Image')
                else:
                    print('Face found in pre upscale check')
                    
            if not do_not_sd_upscale:
                if not use_img2img:
                    last_image_batch = sd_upscale(last_image_batch, scale_factor, is_last,overlap, ui_upscaler_2)
                else: last_image_batch = img2img(last_image_batch, scale_factor, is_last)

            is_last = False

            #print("islast",is_last)

            if face_inpaint_flag:
                for (mask_face, mask_eyes, mask_hair, inpaint_face_size, inpaint_eye_size, face_found, faces_quantity) in mask_create(
                        last_image_batch.images[0],inpaint_hair_flag):
                    if face_found:
                        pos+=1

                        if inpaint_hair_flag:
                            if info_flag:
                                print('Hair inpaint started: ')
                            if not eyes_inpaint_flag and pos == faces_quantity:
                                is_last = True
                            print("islast", is_last)
                            last_image_batch = inpaint(last_image_batch, mask_hair, inpaint_face_size[0],
                                                        inpaint_face_size[1],
                                                        face_denoise,is_last, True)
                            #last_image_batch = image_face_inpaint

                            if inpaint_hair_and_face_flag:

                                if info_flag:
                                    print('Face inpaint started: ')

                                if not eyes_inpaint_flag and pos == faces_quantity:
                                    is_last = True
                                print("islast", is_last)
                                last_image_batch = inpaint(last_image_batch, mask_face, inpaint_face_size[0],
                                                            inpaint_face_size[1],
                                                            face_denoise,is_last, True)
                                
                        else:
                            if info_flag:
                                print('Face inpaint started: ')

                            if not eyes_inpaint_flag and pos == faces_quantity:
                                is_last = True
                            print("islast", is_last)
                            last_image_batch = inpaint(last_image_batch, mask_face, inpaint_face_size[0],
                                                        inpaint_face_size[1],
                                                        face_denoise,is_last, True)

                        if eyes_inpaint_flag:
                            if info_flag:
                                print('Eye inpaint started: ')
                            if pos == faces_quantity:
                                is_last = True
                            print("islast", is_last)
                            last_image_batch = inpaint(last_image_batch, mask_eyes, inpaint_eye_size[0],
                                                        inpaint_eye_size[1], eyes_denoise,is_last, True)
                            #last_image_batch = image_eyes_inpaint
                        #else:
                            #last_image_batch = image_face_inpaint

            #last_image_batch.prompt = initial_seed_and_info[2]
            if i2i_only_flag:
                last_image_batch.outpath_samples = opts.outdir_img2img_samples
            else:
                last_image_batch.outpath_samples = opts.outdir_txt2img_samples
            last_image = last_image_batch.images[0]
            #last_info = last_image_batch.info

            if initial_seed is None:
                initial_seed = last_image_batch.seed
                initial_info = last_image_batch.info

            all_images.append(last_image)
            #all_infos.append(last_info)

            p.seed = last_image_batch.seed + 1
            p.prompt = initial_prompt
            is_last = False
            pos = 0

        processed = Processed(p, all_images, initial_seed, initial_info)
        return processed