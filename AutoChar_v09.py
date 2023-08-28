import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state
from modules import scripts_postprocessing, shared
from modules.processing import StableDiffusionProcessingImg2Img
from modules import processing, shared, sd_samplers, images, devices

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


class Script(scripts.Script):

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):

        return "AutoChar Beta 0.9"

    def ui(self, is_img2img):

        scale_factor = gr.Slider(minimum=1.0, maximum=3.5, step=0.1, value=1.5,
                                 label="Final scale factor")
        

        options = gr.CheckboxGroup(label="Options", choices=['Automatic face inpaint', 'Automatic eyes inpaint',
                                                             'Display info in terminal'],
                                value=['Automatic face inpaint', 'Automatic eyes inpaint'], 
                                elem_id=self.elem_id("options"))



        with gr.Row():
            ui_upscaler_1 = gr.Radio(
                ["Latent (bicubic antialiased)", "Latent", "ESRGAN_4x", "R-ESRGAN 4x+ Anime6B", "R-ESRGAN 4x+", "4x-UltraSharp", "None"], label="Upscaler 1 (High-Res Fix)",
                value="Latent (bicubic antialiased)",

                elem_id=self.elem_id("ui_upscaler_1")
            )



            ui_upscaler_2 = gr.Radio(
                [x.name for x in shared.sd_upscalers], label="Upscaler 2 (SD Upscale)",
                value="Nearest",
                type="index",
                elem_id=self.elem_id("ui_upscaler_2")
            )



        with gr.Accordion('Advanced options', open=False):

            with gr.Row():
                with gr.Column():
                    filtering = gr.Checkbox(True, label="Filtering function")

                    biggest_face = gr.Checkbox(True, label="Inpaint only the biggest face on the image")

                    lower_lora_param = gr.Checkbox(True,
                                            label="Lower LoRA strength for face inpaint. Helps avoid burnout with strong LORAs")
                with gr.Column():
            
                    lower_cfg = gr.Checkbox(False, label="Lower CFG for face inpaint. Helps avoid burning with multiple LoRAs")

                    mid_inpainting = gr.Checkbox(False, label="Attempt mid-uspcale inpainting with chosen options")

                    use_img2img = gr.Checkbox(False, label="Use plain Image2Image instead of SD Upscale")


            with gr.Row():
                with gr.Column():
                    first_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                                                label="High-Res Fix denoising strength")
                    second_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                                                label="SD Upscale denoising strength")
                with gr.Column():
                    face_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.2,
                                                label="Face inpainting denoising strength")
                    eyes_denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.15,
                                                label="Eyes inpainting denoising strength")
                with gr.Column():
                    strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                        label="Strength of Filtering")
                    lora_lowering = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.35,
                                                label="Multiplier for LoRA strength lowering")
            
            scale_factor0 = gr.Slider(minimum=1.0, maximum=1.5, step=0.05, value=1.25,
                    label="High-Res Fix scale factor")
            
            overlap = gr.Slider(label="Tile Overlap parameter for SD Upscale", minimum=0, maximum=256, step=16, value=64, elem_id=self.elem_id("overlap"))


        return [filtering, strength, ui_upscaler_1, first_denoise, second_denoise, face_denoise, eyes_denoise, options,
                mid_inpainting, scale_factor, scale_factor0, lower_cfg,lower_lora_param,biggest_face,ui_upscaler_2,overlap,use_img2img,lora_lowering]

    def run(self, p, filtering, strength, ui_upscaler_1, first_denoise, second_denoise, face_denoise, eyes_denoise,
            options, mid_inpainting, scale_factor, scale_factor0, lower_cfg,lower_lora_param,biggest_face,ui_upscaler_2,overlap,use_img2img,lora_lowering):
        

        initial_seed_and_info = [None, None, None]
        face_inpaint_flag = True if "Automatic face inpaint" in options else False
        eyes_inpaint_flag = True if "Automatic eyes inpaint" in options else False
        info_flag = True if "Display info in terminal" in options else False
        mid_face_flag = None
        mid_eyes_flag = None

        if info_flag:
            print(options)
            print('filtering', filtering, 'strength', strength, 'ui_upscaler_1', ui_upscaler_1, 'face_inpaint_flag',
                  face_inpaint_flag, 'eyes_inpaint_flag', eyes_inpaint_flag, 'scale_factor', scale_factor)

        if mid_inpainting:
            mid_face_flag = face_inpaint_flag
            mid_eyes_flag = eyes_inpaint_flag

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
        state.job_count = batch_count

        instance_img2img = StableDiffusionProcessingImg2Img()
        instance_img2img.outpath_samples = opts.outdir_img2img_samples
        instance_inpaint = StableDiffusionProcessingImg2Img()
        instance_inpaint.outpath_samples = opts.outdir_img2img_samples
        instance_sd_upscale = StableDiffusionProcessingImg2Img()
        instance_sd_upscale.outpath_samples = opts.outdir_img2img_samples

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
        def mask_create(image):

            #  Regulating parameters
            face_resolution_scale = 2.5
            resolution_scale = 1.5
            mask_dilation = 1.5
            face_found = True
            rotate = False
            directory = os.path.dirname(__file__)
            image = np.array(image)

            # Load the model
            try:
                weights = os.path.join(directory, "face_detection_yunet_2023mar.onnx")
            except:
                weights = os.path.join(directory, "face_detection_yunet_2022mar.onnx")
                
            face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

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
                    _, faces = face_detector.detect(image)
                    print(faces)

            results = []
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
                box = [box[0], box[1], closest(int(box[2] * 1.25), 4),
                       closest(int(box[3] * face_height_multiplier), 4)]
                cv2.rectangle(mask1, box, 255, -1)
                cv2.rectangle(mask2, eye_box, 255, -1)
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
                    (Image.fromarray(mask1), Image.fromarray(mask2), inpaint_face_size, inpaint_eye_size, face_found, faces_quantity))


            return results

        # Function for lowering LORA strength
        def lower_lora(lora_list_str):
            new_lora_list = []
            lora_f_list = lora_list_str.split(" ")
            for lora in lora_f_list:
                lora_strengh = round(((float((lora.split(":")[2])[:-1])) * lora_lowering),3)
                new_lora_str = lora.split(":")[0] + ':' + lora.split(":")[1] + ':' + str(lora_strengh) + '>'
                new_lora_list.append(new_lora_str)
            final_lora_str = ' '.join([str(elem) for i, elem in enumerate(new_lora_list)])
            return final_lora_str


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
            p.hr_second_pass_steps = 10

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
            instance_img2img.seed = init_image.seed
            instance_img2img.init_images = [init_image.images[0]]
            instance_img2img.denoising_strength = second_denoise
            instance_img2img.steps = 12

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

            # Clear cache
            torch.cuda.empty_cache()

            # Check if it's our last step
            if is_last_sd_upscale:
                instance_img2img.outpath_samples = opts.outdir_txt2img_samples


            # Check for changes in prompt (DYNAMIC PROMPTS PROBLEM)
            if prompt_temp != initial_prompt:
                instance_sd_upscale.prompt = prompt_temp

            # Change parameters
            instance_sd_upscale.negative_prompt = init_image.negative_prompt
            instance_sd_upscale.seed = init_image.seed
            instance_sd_upscale.init_images = [init_image.images[0]]
            instance_sd_upscale.extra_generation_params["SD upscale overlap"] = 64
            instance_sd_upscale.extra_generation_params["SD upscale upscaler"] = shared.sd_upscalers[ui_upscaler_2_func].name
            instance_sd_upscale.steps = 12
            instance_sd_upscale.denoising_strength = second_denoise
            instance_sd_upscale.do_not_save_grid = True
            instance_sd_upscale.do_not_save_samples = True

            # Change resolution so that it's surely dividable by 4
            instance_sd_upscale.width = closest(int(scale_factor * p.width), 4)
            #instance_sd_upscale.height = closest(int(scale_factor * p.height), 4)

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

            # Change parameters
            if not rewrite_seed:
                instance_inpaint.seed = init_image.seed
            else:
                instance_inpaint.seed = init_image.seed + 10000

            # Check for changes in prompt (DYNAMIC PROMPTS PROBLEM)
            if prompt_temp != initial_prompt:
                instance_inpaint.prompt = prompt_temp

            if lower_cfg:
                if instance_inpaint.cfg_scale != p.cfg_scale:
                    instance_inpaint.cfg_scale = p.cfg_scale
                instance_inpaint.cfg_scale = int(instance_inpaint.cfg_scale * 0.66)
                if info_flag:
                    print('Lowering CFG for inpaint \n new CFG:', instance_inpaint.cfg_scale)

            if lower_lora_param:
                new_prompt = re.split('<lora:', instance_inpaint.prompt)[0] + lower_lora(loras_list)
                instance_inpaint.prompt = new_prompt
                #print('New prompt after lowering LORAS', new_prompt)
                #print(instance_inpaint.prompt)
                if info_flag:
                    print('Lowering LORA strength for inpaint \n new LORA strengths:', lower_lora(loras_list))


            # Check if it's our last step
            if is_last_inpaint:
                instance_inpaint.outpath_samples = opts.outdir_txt2img_samples

            instance_inpaint.negative_prompt = init_image.negative_prompt
            instance_inpaint.seed = init_image.seed
            instance_inpaint.init_images = [init_image.images[0]]
            instance_inpaint.image_mask = p_mask
            instance_inpaint.mask_blur = int(w * 0.01)
            instance_inpaint.inpainting_fill = 1
            instance_inpaint.inpaint_full_res = True
            instance_inpaint.denoising_strength = denoise
            instance_inpaint.steps = 12
            instance_inpaint.width = w
            instance_inpaint.height = h

            # Print inpaint parameters
            if info_flag:
                print('Inpaint parameters: ', instance_inpaint.steps, instance_inpaint.width, instance_inpaint.height)

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
            state.job = f"batch {n} of {batch_count} batches \n"
            print(state.job)
            hr_fix_output = text2img_hr(upscaler, scale_factor0)
            #hf_fix_output_str = ' '.join([str(elem) for i, elem in enumerate(hr_fix_output.info)])
            #prompt_temp = hr_fix_output.info
           # prompt_temp = hr_fix_output.info.split(('egative prompt:')[0])
            prompt_temp = re.split('Negative prompt: ',hr_fix_output.info)[0]
            #print(prompt_temp)
            #print (initial_prompt)
            loras = re.findall(r'<.*?>', initial_prompt)
            loras_list = ' '.join([str(elem) for i, elem in enumerate(loras)])
            #print ("Used LORAs " +str(loras_list))
            if filtering:
                hr_fix_output.images[0] = enhance_image(hr_fix_output, strength)

            if mid_face_flag:
                for (mask_face, mask_eyes, inpaint_face_size, inpaint_eye_size, face_found, faces_quantity) in mask_create(
                        hr_fix_output.images[0]):
                    if face_found:
                        if info_flag:
                            print('Mid-uspcale face inpaint started: ')

                        mid_image_face_inpaint = inpaint(hr_fix_output, mask_face, inpaint_face_size[0],
                                                         inpaint_face_size[1],
                                                         face_denoise,is_last, True)

                        if mid_eyes_flag:
                            if info_flag:
                                print('Mid-uspcale eyes inpaint started: ')
                            mid_image_eyes_inpaint = inpaint(mid_image_face_inpaint, mask_eyes, inpaint_eye_size[0],
                                                             inpaint_eye_size[1], eyes_denoise,is_last, True)
                            hr_fix_output.images[0] = mid_image_eyes_inpaint.images[0]
                        else:
                            hr_fix_output.images[0] = mid_image_face_inpaint.images[0]


            if not face_inpaint_flag:
                is_last = True

            if not use_img2img:
                last_image_batch = sd_upscale(hr_fix_output, scale_factor, is_last,overlap, ui_upscaler_2)
            else: last_image_batch = img2img(hr_fix_output, scale_factor, is_last)

            print("islast",is_last)
            if face_inpaint_flag:
                for (mask_face, mask_eyes, inpaint_face_size, inpaint_eye_size, face_found, faces_quantity) in mask_create(
                        last_image_batch.images[0]):
                    if face_found:
                        pos+=1
                        if info_flag:
                            print('Face inpaint started: ')

                        if not eyes_inpaint_flag and pos == faces_quantity:
                            is_last = True
                        print("islast", is_last)
                        image_face_inpaint = inpaint(last_image_batch, mask_face, inpaint_face_size[0],
                                                     inpaint_face_size[1],
                                                     face_denoise,is_last, True)

                        if eyes_inpaint_flag:
                            if info_flag:
                                print('Eye inpaint started: ')
                            if pos == faces_quantity:
                                is_last = True
                            print("islast", is_last)
                            image_eyes_inpaint = inpaint(image_face_inpaint, mask_eyes, inpaint_eye_size[0],
                                                         inpaint_eye_size[1], eyes_denoise,is_last, True)
                            last_image_batch = image_eyes_inpaint
                        else:
                            last_image_batch = image_face_inpaint

            #last_image_batch.prompt = initial_seed_and_info[2]
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
