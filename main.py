import gradio as gr

from config.registers import params_models, upscalers
from config.schemas import UpscalerMethodEnum

choices = [upscaler.value for upscaler in UpscalerMethodEnum]


# Define the function that will be called when the button is clicked
# You need to define the inputs with the same name as the inputs defined in the UI
# just for the sake of clarity
def upscaler_factory(
        image,
        upscaler: UpscalerMethodEnum,
        scale,
        num_inference_steps,
        eta,
        prompt,
        num_steps,
        guidance_scale,
):
    # build params dict
    params = locals()
    params.pop('image')

    # get upscaler object
    pipeline_object = upscalers[params['upscaler']]
    # get params object
    params_object_builder = params_models.get(params['upscaler'])
    # build params object according to the upscaler
    upscaler_params = params_object_builder(**params)
    # init upscaler
    pipe = pipeline_object(params=upscaler_params)
    # upscale image
    upscaled_image = pipe.generate(image=image)
    return upscaled_image


def set_visible_params_by_upscaler(upscaler):
    if upscaler == UpscalerMethodEnum.REALESRGAN:
        return (gr.Radio(label="Scale", choices=[2,4], interactive=False, value=4),
                gr.Group(visible=False),
                gr.Group(visible=False)
                )
    elif upscaler == UpscalerMethodEnum.LDMSUPERRESOL4X:
        return (gr.Radio(label="Scale", choices=[2,4], interactive=False, value=4),
                gr.Group(visible=True),
                gr.Group(visible=False)
                )
    elif upscaler == UpscalerMethodEnum.SD4X:
        return (gr.Radio(label="Scale", choices=[2,4], interactive=False, value=4),
                gr.Group(visible=False),
                gr.Group(visible=True)
                )
    return (gr.Radio(label="Scale", choices=[2, 4], interactive=True, value=4),
            gr.Group(visible=False),
            gr.Group(visible=False)
            )


with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("## Upscaling demo")
    with gr.Row():
        with gr.Column():
            initial_image = gr.Image(type='pil')
            upscaler = gr.Radio(label="Upscaling method", choices=choices, value=choices[0])
            scale = gr.Radio(label="Scale", choices=[2, 4], interactive=True, value=4)
            with gr.Group(visible=False) as ldm_model:
                gr.Markdown(
                    '''
                    ####  &nbsp; Additional parameters
                    '''
                )
                num_inference_steps = gr.Slider(label='number of inference steps', value=1, interactive=True,
                                                   minimum=1, maximum=200)
                eta = gr.Number(label='eta', value=1, interactive=True)
            with gr.Group(visible=False) as sd_model:
                gr.Markdown(
                    '''
                    ####  &nbsp; Additional parameters
                    '''
                )
                prompt = gr.Textbox(label='Prompt', placeholder="Enter a prompt to guide the upscaling",
                                    lines=3, interactive=True)
                num_steps = gr.Slider(label='number of inference steps', value=20, interactive=True,
                                                   minimum=1, maximum=200)
                guidance_scale = gr.Slider(label='guidance scale', value=0, interactive=True,
                                                   minimum=0, maximum=20)

        with gr.Column(scale=2):
            output = gr.Image(height=512)
            with gr.Row():
                clear_btn = gr.ClearButton(components=[output])
                upscale_btn = gr.Button("Upscale")

    upscaler.change(
        fn=set_visible_params_by_upscaler,
        inputs=upscaler,
        outputs=[scale, ldm_model, sd_model]
    )

    # Define all inpunt for the UI independent of the model chosen
    inputs = [
        initial_image,
        upscaler,
        scale,
        num_inference_steps,
        eta,
        prompt,
        num_steps,
        guidance_scale
    ]

    upscale_btn.click(fn=upscaler_factory, inputs=inputs, outputs=[output])

demo.launch(show_error=True, share=True)
