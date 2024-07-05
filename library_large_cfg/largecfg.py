
class ThresholdChangingNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "initial_cfg": ("FLOAT", {"default": 11.5, "min": 0.0, "max": 100.0, "step": 0.5}),
                "stop_at" : ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/initialcfg"

    def patch(self, model, initial_cfg, stop_at):

        def sampler_cfg(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            #orig_result  = uncond + (cond - uncond) * cond_scale
            time_step = model.model.model_sampling.timestep(args["sigma"])
            time_step = time_step[0].item()
            stop_time_step = int(999 * stop_at)
            #print(f"Time step: {time_step}, Stop time step: {stop_time_step} apply initial_cfg: {time_step < (999 - stop_time_step)} , cfg to apply: {cond_scale if time_step < (999 - stop_time_step) else initial_cfg}")
            cond_to_use = cond_scale if time_step < (999 - stop_time_step) else initial_cfg
            return input - (uncond + (cond - uncond) * cond_to_use)

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_cfg)
        return (m, )
