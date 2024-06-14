import torch
import os
import json
import safetensors.torch

import comfy.samplers
import execution
import server

SAMPLER_NODES = ["SamplerCustom", "KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"]

def store_checkpoint(unique_id, tensors, metadata, priority=0):
    """Swappable interface for saving checkpoints.
       Implementation must be transactional: Either the whole thing completes,
       or the prior checkpoint must be valid even if crash occurs mid execution"""
    file =  f"checkpoint/{unique_id}.checkpoint"
    safetensors.torch.save_file(tensors, file, metadata)
def get_checkpoint(unique_id):
    """Returns the information previously saved"""
    file = f"checkpoint/{unique_id}.checkpoint"
    if not os.path.exists(file):
        return None, None
    with safetensors.torch.safe_open(file, framework='pt' ) as f:
        metadata = f.metadata()
        tensors = {key:f.get_tensor(key) for key in f.keys()}
        return tensors, metadata
def reset_checkpoints(unique_id=None):
    """Clear all checkpoint information."""
    if unique_id is not None:
        if os.path.exists(f"checkpoint/{unique_id}.checkpoint"):
            os.remove(f"checkpoint/{unique_id}.checkpoint")
        return
    for file in os.listdir("checkpoint"):
        os.remove(os.path.join("checkpoint", file))

class CheckpointSampler(comfy.samplers.KSAMPLER):
    def sample(self, *args, **kwargs):
        args = list(args)
        self.unique_id = server.PromptServer.instance.last_node_id
        self.step = None
        data, metadata = get_checkpoint(self.unique_id)
        if metadata is not None and 'step' in metadata:
            data = data['x']
            self.step = int(metadata['step'])
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
        data = safetensors.torch.save
        store_checkpoint(self.unique_id, {'x':x}, {'step':str(step)})

original_recursive_execute = execution.recursive_execute
def recursive_execute_injection(*args):

    unique_id = args[3]
    class_type = args[1][unique_id]['class_type']
    #Imperfect, is checked for each bubble down step
    #Only applied once, but has unnecessary loads
    if len(args[5]) == 0:
        metadata = get_checkpoint('prompt')[1]
        if metadata is None or json.loads(metadata['prompt']) != args[1]:
            reset_checkpoints()
            store_checkpoint('prompt', {'x': torch.ones(1)},
                             {'prompt': json.dumps(args[1])}, priority=2)
    if  class_type in SAMPLER_NODES:
        data, metadata = get_checkpoint(unique_id)
        if metadata is not None and 'step' in metadata:
            args[1][unique_id]['inputs']['latent_image'] = ['checkpointed'+unique_id, 0]
            args[2]['checkpointed'+unique_id] = [[{'samples': data['x']}]]
        elif metadata is not None and 'completed' in metadata:
            outputs = json.loads(metadata['completed'])
            for x in range(len(outputs)):
                if outputs[x] == 'tensor':
                    outputs[x] = list(data[str(x)])
                elif outputs[x] == 'latent':
                    outputs[x] = [{'samples': l} for l in data[str(x)]]
            args[2][unique_id] = outputs
            return True, None, None

    res = original_recursive_execute(*args)
    #Conditionally save node output
    #TODO: determine which non-sampler nodes are worth saving
    if class_type in SAMPLER_NODES and unique_id in args[2]:
        data = {}
        outputs = args[2][unique_id].copy()
        for x in range(len(outputs)):
            if isinstance(outputs[x][0], torch.Tensor):
                data[str(x)] = torch.stack(outputs[x])
                outputs[x] = 'tensor'
            elif isinstance(outputs[x][0], dict):
                data[str(x)] = torch.stack([l['samples'] for l in outputs[x]])
                outputs[x] = 'latent'
        store_checkpoint(unique_id, data, {'completed': json.dumps(outputs)}, priority=1)
    return res

comfy.samplers.KSAMPLER = CheckpointSampler
execution.recursive_execute = recursive_execute_injection

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
