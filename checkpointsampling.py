import torch
import os
import json
import safetensors

import comfy.samplers
import execution
import server

class CheckpointSampler(comfy.samplers.KSAMPLER):
    def sample(self, *args, **kwargs):
        args = list(args)
        self.unique_id = server.PromptServer.instance.last_node_id
        step = None
        with open(f"checkpoint/{self.unique_id}.json", "r") as f:
            f.readline()
            while line := f.readline():
                step = int(line)
        if step is not None:
            #checkpoint of execution exists
            args[5] = safetensors.torch.load_file(f"checkpoint/{self.unique_id}.{step%2}.latent")['latent_tensor'].to(args[4].device)
            args[1] = args[1][step:]
            #disable added noise, as the checkpointed latent is already noised
            args[4][:] = 0
        original_callback = kwargs.get("callback", None)
        def callback(*args):
            self.callback(args)
            if original_callback is not None:
                return original_callback(*args)
        args[3] = callback
        res = super().sample(*args, **kwargs)
        #cleanup checkpoints as execution has completed
        #TODO: Have tracking for freshness and keep checkpoints post execution?
        if os.path.exists(f"checkpoint/{self.unique_id}.0.latent"):
            os.remove(f"checkpoint/{self.unique_id}.0.latent")
        if os.path.exists(f"checkpoint/{self.unique_id}.1.latent"):
            os.remove(f"checkpoint/{self.unique_id}.1.latent")
        if os.path.exists(f"checkpoint/{self.unique_id}.json"):
            os.remove(f"checkpoint/{self.unique_id}.json")
        return res

    def callback(self, args):
        x = args[2]
        print(x.sum())
        step = args[0]
        comfy.utils.save_torch_file({"latent_tensor": x}, f"checkpoint/{self.unique_id}.{step%2}.latent")
        with open(f"checkpoint/{self.unique_id}.json", "a") as f:
            f.write("\n"+str(step))
        if step == 3:
            if args[3] == 10:
                raise Exception("test crash")
            #manual exception raising
original_recursive_execute = execution.recursive_execute
def recursive_execute_injection(*args):
    unique_id = args[3]
    prev_prompt = None
    if args[1][unique_id]['class_type'] == "SamplerCustom":
        if os.path.exists(f"checkpoint/{unique_id}.json"):
            with open(f"checkpoint/{unique_id}.json", "r") as f:
                prev_prompt = json.loads(f.readline())
                if f.readline() == '':
                    #execution interrupted before checkpoint made
                    prev_prompt = None
            #TODO: double check this is deep compare
        if prev_prompt == args[1]:
            #NOTE: to avoid precision lost on rescaling/ease of implementation,
            #the tensor is loaded twice, this first load is just for dimensions,
            #so the actual index doesn't matter
            x = safetensors.torch.load_file(f"checkpoint/{unique_id}.0.latent")['latent_tensor']
            args[1][unique_id]['inputs']['latent_image'] = ['checkpointed'+unique_id, 0]
            args[2]['checkpointed'+unique_id] = [[{'samples': x}]]
        else:
            with open(f"checkpoint/{unique_id}.json", "w") as f:
                json.dump(args[1],f)
    return original_recursive_execute(*args)



comfy.samplers.KSAMPLER = CheckpointSampler
execution.recursive_execute = recursive_execute_injection

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
