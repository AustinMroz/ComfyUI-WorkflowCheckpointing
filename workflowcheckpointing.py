import torch
import os
import json
import safetensors.torch
import aiohttp
import asyncio
import queue
import threading
import logging

import comfy.samplers
import execution
import server

SAMPLER_NODES = ["SamplerCustom", "KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"]

class RequestLoop:
    def __init__(self):
        self.active_request = None
        self.current_priority = 0
        self.queue_high = queue.Queue()
        self.low = None
        self.mutex = threading.RLock()
        #main.py has already created an event loop
        event_loop = server.PromptServer.instance.loop
        self.process_loop = event_loop.create_task(self.process_requests())
    def queue(self, req, prio):
        with self.mutex:
            if prio == 2:
                self.low = None
                self.queue_high = queue.Queue()
                self.queue_high.put(req)
                if self.active_request is not None:
                    pass
                    #self.process_loop.cancel()
            elif prio == 1:
                self.low = None
                self.queue_high.put(req)
                if self.current_priority == 0:
                    pass
                    #self.process_loop.cancel()
            else:
                self.low = req
    async def process_requests(self):
        async with aiohttp.ClientSession() as session:
            async with session.get('http://169.254.169.254:80/v1/token') as r:
                token = await r.json()['jwt']
        headers = {'Authorization': token}
        async with aiohttp.ClientSession('https://storage-api.salad.com') as session:
            while True:
                if self.active_request is None:
                    await asyncio.sleep(.1)
                else:
                    try:
                        req = self.active_request
                        async with session.put(req[0], headers=headers, **req[1]) as r:
                            #We don't care about result, but must still await it
                            await r.text()
                    except asyncio.CancelledError:
                        #TODO, ensure we only swallow our tasks?
                        pass
                with self.mutex:
                    if not self.queue_high.empty():
                        self.active_request = self.queue_high.get()
                    else:
                        if self.low is not None:
                            self.active_request = self.low
                            self.low = None
                        else:
                            self.active_request = None

#Placeholder, would need to be pulled from salad
ORGANIZATION = "banodoco"
MACHINEID = "local"

class NetCheckpoint:
    def __init__(self):
        self.requestloop = RequestLoop()
        self.has_warned_size = False
    def store(self, unique_id, tensors, metadata, priority=0):
        file = "/".join(['', ORGANIZATION, MACHINEID, "checkpoint", f"{unique_id}.checkpoint"])
        data = safetensors.torch.save(tensors, metadata)
        if len(data) > 10 ** 8:
            if not self.has_warned_size:
                logging.warning("Checkpoint is too large and has been skipped")
                self.has_warned_size = True
            return
        self.requestloop.queue((file, {'data': data}), priority)
    def get(self, unique_id):
        """Returns the information previously saved
           If the request has checkpointed data, this information should
           be loaded prior to job start"""
        file = f"input/checkpoint/{unique_id}.checkpoint"
        if not os.path.exists(file):
            return None, None
        with safetensors.torch.safe_open(file, framework='pt' ) as f:
            metadata = f.metadata()
            tensors = {key:f.get_tensor(key) for key in f.keys()}
            return tensors, metadata
    def reset(self, unique_id=None):
        """Clear all checkpoint information."""
        if unique_id is not None:
            if os.path.exists(f"input/checkpoint/{unique_id}.checkpoint"):
                os.remove(f"input/checkpoint/{unique_id}.checkpoint")
            return
        for file in os.listdir("input/checkpoint"):
            os.remove(os.path.join("input/checkpoint", file))

class FileCheckpoint:
    def store(self, unique_id, tensors, metadata, priority=0):
        """Swappable interface for saving checkpoints.
           Implementation must be transactional: Either the whole thing completes,
           or the prior checkpoint must be valid even if crash occurs mid execution"""
        file =  f"checkpoint/{unique_id}.checkpoint"
        safetensors.torch.save_file(tensors, file, metadata)
    def get(self, unique_id):
        """Returns the information previously saved"""
        file = f"checkpoint/{unique_id}.checkpoint"
        if not os.path.exists(file):
            return None, None
        with safetensors.torch.safe_open(file, framework='pt' ) as f:
            metadata = f.metadata()
            tensors = {key:f.get_tensor(key) for key in f.keys()}
            return tensors, metadata
    def reset(self, unique_id=None):
        """Clear all checkpoint information."""
        if unique_id is not None:
            if os.path.exists(f"checkpoint/{unique_id}.checkpoint"):
                os.remove(f"checkpoint/{unique_id}.checkpoint")
            return
        for file in os.listdir("checkpoint"):
            os.remove(os.path.join("checkpoint", file))

checkpoint = NetCheckpoint() if "USE_NET_CHECKPOINTING" in os.environ else FileCheckpoint()

class CheckpointSampler(comfy.samplers.KSAMPLER):
    def sample(self, *args, **kwargs):
        args = list(args)
        self.unique_id = server.PromptServer.instance.last_node_id
        self.step = None
        data, metadata = checkpoint.get(self.unique_id)
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
        return res

    def callback(self, step, denoised, x, total_steps):
        if self.step is not None:
            step += self.step
        data = safetensors.torch.save
        checkpoint.store(self.unique_id, {'x':x}, {'step':str(step)})

original_recursive_execute = execution.recursive_execute
def recursive_execute_injection(*args):

    unique_id = args[3]
    class_type = args[1][unique_id]['class_type']
    #Imperfect, is checked for each bubble down step
    #Only applied once, but has unnecessary loads
    if len(args[5]) == 0:
        metadata = checkpoint.get('prompt')[1]
        if metadata is None or json.loads(metadata['prompt']) != args[1]:
            checkpoint.reset()
            checkpoint.store('prompt', {'x': torch.ones(1)},
                             {'prompt': json.dumps(args[1])}, priority=2)
    if  class_type in SAMPLER_NODES:
        data, metadata = checkpoint.get(unique_id)
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
        checkpoint.store(unique_id, data, {'completed': json.dumps(outputs)}, priority=1)
    return res

comfy.samplers.KSAMPLER = CheckpointSampler
execution.recursive_execute = recursive_execute_injection

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
