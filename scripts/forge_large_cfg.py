import gradio as gr

from modules import scripts
from library_large_cfg.largecfg import ThresholdChangingNode

comfyLargeCfgNode = ThresholdChangingNode().patch


class LargeInitialThresholdingForForge(scripts.Script):
    sorting_priority = 9

    def title(self):
        return "Large Initial Thresholding for Forge"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            initial_cfg = gr.Slider(label='Initial CFG', minimum=0.0, maximum=100.0, step=0.5, value=7.0)
            stop_percentile = gr.Slider(label='Stop at step Percentile', minimum=0.0, maximum=1.0, step=0.01,
                                             value=0.05)

        return enabled, initial_cfg, stop_percentile

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, initial_cfg, stop_percentile = script_args

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = comfyLargeCfgNode(unet, initial_cfg, stop_percentile)[0]

        p.sd_model.forge_objects.unet = unet
        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            initial_cfg=initial_cfg,
            stop_percentile=stop_percentile
        ))

        return
