import torch
import os
import json
import safetensors

import comfy.samplers
import execution
import server

SAMPLER_NODES = ["SamplerCustom", "KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"]

def store_checkpoint(unique_id, data, partial_progress_counter=-2, is_tensor=True):
    """Swappable interface for saving checkpoints.
       Implementation must be transactional: Either the whole thing completes,
       or the prior checkpoint must be valid even if crash occurs mid execution"""
    file =  f"checkpoint/{unique_id}.{partial_progress_counter%2}.safetensor"
    if is_tensor:
        comfy.utils.save_torch_file({'x': data},file)
    else:
        with open(file, "w") as f:
            json.dump(data,f)
    with open(f"checkpoint/{unique_id}.txt", "w") as f:
        f.write(str(partial_progress_counter)+"\n"+str(1*is_tensor))
def get_checkpoint(unique_id):
    """Returns the information previously saved"""
    partial_progress = -2
    is_tensor = True
    if os.path.exists(f"checkpoint/{unique_id}.txt"):
        with open(f"checkpoint/{unique_id}.txt", "r") as f:
            partial_progress = int(f.readline())
            is_tensor = f.readline() != '0'
    file = f"checkpoint/{unique_id}.{partial_progress%2}.safetensor"
    if not os.path.exists(file):
        return None, None
    if is_tensor:
        res = safetensors.torch.load_file(file)['x']
    else:
        with open(file, "r") as f:
            res = json.load(f)
    return partial_progress, res
def reset_checkpoints(unique_id=None):
    """Clear all checkpoint information."""
    if unique_id is not None:
        if os.path.exists(f"checkpoint/{unique_id}.0.safetensor"):
            os.remove(f"checkpoint/{unique_id}.0.safetensor")
        if os.path.exists(f"checkpoint/{unique_id}.1.safetensor"):
            os.remove(f"checkpoint/{unique_id}.1.safetensor")
        if os.path.exists(f"checkpoint/{unique_id}.txt"):
            os.remove(f"checkpoint/{unique_id}.txt")
        return
    for file in os.listdir("checkpoint"):
        os.remove(os.path.join("checkpoint", file))

class CheckpointSampler(comfy.samplers.KSAMPLER):
    def sample(self, *args, **kwargs):
        args = list(args)
        self.unique_id = server.PromptServer.instance.last_node_id
        self.step, data = get_checkpoint(self.unique_id)
        if self.step is not None:
            #checkpoint of execution exists
            args[5] = data.to(args[4].device)
            args[1] = args[1][self.step:]
            #disable added noise, as the checkpointed latent is already noised
            args[4][:] = 0
        original_callback = args[3]
        def callback(*args):
            self.callback(*args)
            if original_callback is not None:
                return original_callback(*args)
        args[3] = callback
        res = super().sample(*args, **kwargs)
        reset_checkpoints(self.unique_id)
        return res

    def callback(self, step, denoised, x, total_steps):
        if self.step is not None:
            step += self.step
        store_checkpoint(self.unique_id, x, step)

original_recursive_execute = execution.recursive_execute
def recursive_execute_injection(*args):

    unique_id = args[3]
    class_type = args[1][unique_id]['class_type']
    if len(args[5]) == 0:
        _, prev_prompt = get_checkpoint('prompt')
        if prev_prompt != args[1]:
            reset_checkpoints()
            store_checkpoint('prompt', args[1], is_tensor=False)
    if  class_type in SAMPLER_NODES:
        step, x = get_checkpoint(unique_id)
        if step is not None and step>0:
            args[1][unique_id]['inputs']['latent_image'] = ['checkpointed'+unique_id, 0]
            args[2]['checkpointed'+unique_id] = [[{'samples': x}]]
    res = original_recursive_execute(*args)
    #Conditionally save node output
    #TODO: determine which non-sampler nodes are worth saving
    if class_type in SAMPLER_NODES:
        pass
        #output = args[2][unique_id]
    return res

comfy.samplers.KSAMPLER = CheckpointSampler
execution.recursive_execute = recursive_execute_injection

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
