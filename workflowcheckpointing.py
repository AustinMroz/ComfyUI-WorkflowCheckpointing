import torch
import os
import json
import safetensors.torch
import aiohttp
import asyncio
import queue
import threading
import logging
import itertools
import hashlib

import comfy.samplers
import execution
import server

SAMPLER_NODES = ["SamplerCustom", "KSampler", "KSamplerAdvanced", "SamplerCustomAdvanced"]

SALAD_TOKEN = None
async def get_header():
    if 'SALAD_API_KEY' in os.environ:
        #NOTE: Only for local testing. Do not add to container
        return {'Salad-Api-Key': os.environ['SALAD_API_KEY']}
    global SALAD_TOKEN
    if SALAD_TOKEN is None:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://169.254.169.254:80/v1/token') as r:
                SALAD_TOKEN =(await r.json())['jwt']
    return {'Authorization': SALAD_TOKEN}

class RequestLoop:
    def __init__(self):
        self.active_request = None
        self.current_priority = 0
        self.queue_high = queue.Queue()
        self.low = None
        self.mutex = threading.RLock()
        self.do_reset = False
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
    def reset(self, uid):
        with self.mutex:
            self.low = None
            self.queue_high = queue.Queue()
            self.do_reset = uid
    async def delete_file(self, s, url, semaphore):
        async with semaphore:
            async with s.delete(url, headers=await get_header()) as r:
                await r.text()
    async def _reset(self, s, uid):
        base_url = '/organizations/' + ORGANIZATION +'/files'
        checkpoint_base = '/'.join([base_url, uid, 'checkpoint'])
        checkpoint_base = 'https://storage-api.salad.com' + checkpoint_base
        async with s.get(base_url, headers=await get_header()) as r:
            js = await r.json()
            files = js['files']
        checkpoints =  list(filter(lambda x: x['url'].startswith(checkpoint_base), files))
        for cp in checkpoints:
            cp['url'] = cp['url'][29:]
        semaphore = asyncio.Semaphore(5)
        deletes = [asyncio.create_task(self.delete_file(s, f['url'], semaphore)) for f in checkpoints]
        if len(deletes) > 0:
            await asyncio.gather(*deletes)
    async def process_requests(self):
        headers = await get_header()
        async with aiohttp.ClientSession('https://storage-api.salad.com') as session:
            while True:
                if self.do_reset != False:
                    await self._reset(session, self.do_reset)
                    self.do_reset = False
                if self.active_request is None:
                    await asyncio.sleep(.1)
                else:
                    req = self.active_request
                    fd = aiohttp.FormData({'file': req[1]})
                    async with session.put(req[0], headers=headers, data=fd) as r:

                        #We don't care about result, but must still await it
                        await r.text()
                with self.mutex:
                    if not self.queue_high.empty():
                        self.active_request = self.queue_high.get()
                    else:
                        if self.low is not None:
                            self.active_request = self.low
                            self.low = None
                        else:
                            self.active_request = None

ORGANIZATION = os.environ.get('SALAD_ORGANIZATION', None)
class NetCheckpoint:
    def __init__(self):
        self.requestloop = RequestLoop()
        self.has_warned_size = False
        assert ORGANIZATION is not None
    def store(self, unique_id, tensors, metadata, priority=0):
        file = "/" + "/".join(['organizations', ORGANIZATION, 'files', self.uid,
                         "checkpoint", f"{unique_id}.checkpoint"])
        data = safetensors.torch.save(tensors, metadata)
        if len(data) > 10 ** 8:
            if not self.has_warned_size:
                logging.warning("Checkpoint is too large and has been skipped")
                self.has_warned_size = True
            return
        self.requestloop.queue((file, data), priority)
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
        #TODO: filter delete requests by node uniqueid
        """Clear all checkpoint information."""
        self.requestloop.reset(self.uid)
        if unique_id is not None:
            if os.path.exists(f"input/checkpoint/{unique_id}.checkpoint"):
                os.remove(f"input/checkpoint/{unique_id}.checkpoint")
            return
        os.makedirs("input/checkpoint", exist_ok=True)
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

checkpoint = NetCheckpoint() if "SALAD_ORGANIZATION" in os.environ else FileCheckpoint()

def file_hash(filename):
    h = hashlib.sha256()
    b = bytearray(10*1024*1024) # read 10 megabytes at a time
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(b):
            h.update(b)
    return h.hexdigest()

async def fetch_remote_file(session, file, semaphore):
    filename = os.path.join("input", file['filepath'])
    assert filename.find("..") == -1, "Paths may not contain .."
    if os.path.exists(filename) and 'hash' in file and file_hash(filename) == file['hash']:
        return
    if file['url'].startswith('https://storage-api.salad.com/'):
        headers = await get_header()
    else:
        headers = {}
    async with semaphore:
        async with session.get(file['url'], headers=headers) as r:
            with open(filename, 'wb') as fd:
                async for chunk in r.content.iter_chunked(2**16):
                    fd.write(chunk)

async def fetch_remote_files(remote_files, uid=None):
    #TODO: Add requested support for zip files?
    async with aiohttp.ClientSession() as s:
        base_url = 'https://storage-api.salad.com/organizations/' + ORGANIZATION +'/files'
        if uid is not None:
            checkpoint_base = '/'.join([base_url, uid, 'checkpoint'])
            async with s.get(base_url, headers=await get_header()) as r:
                js = await r.json()
                files = js['files']
            checkpoints =  list(filter(lambda x: x['url'].startswith(checkpoint_base), files))
            for cp in checkpoints:
                cp['filepath'] = os.path.join('checkpoint',
                                              cp['url'][len(checkpoint_base)+1:])
            remote_files = itertools.chain(remote_files, checkpoints)
        semaphore = asyncio.Semaphore(5)
        fetches = [asyncio.create_task(fetch_remote_file(s, f, semaphore)) for f in remote_files]
        if len(fetches) > 0:
            await asyncio.gather(*fetches)

@server.PromptServer.instance.routes.post("/prompt_remote")
async def post_prompt(request):
    json_data = await request.json()
    if "prompt" in json_data and "extra_data" in json_data and "SALAD_ORGANIZATION" in os.environ:
        extra_data = json_data["extra_data"]
        #NOTE: Rendered obsolete by existing infrastructure, can be pruned
        remote_files = extra_data.get("remote_files", [])
        uid = extra_data.get("uid", 'local')
        checkpoint.uid = uid
        await fetch_remote_files(remote_files, uid=uid)
    #probably a bad idea
    return await server.PromptServer.instance.routes[15].handler(request)

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
        if self.step is None and "FORCE_CRASH_AT" in os.environ:
            if step == int(os.environ['FORCE_CRASH_AT']):
                raise Exception("Simulated Crash")

original_recursive_execute = execution.recursive_execute
def recursive_execute_injection(*args):

    unique_id = args[3]
    class_type = args[1][unique_id]['class_type']
    extra_data = args[4]
    if 'checkpoints' in extra_data:
        checkpoint.update(extra_data.pop('checkpoints'))
    if 'prompt_checked' not in args[4]:
        metadata = checkpoint.get('prompt')[1]
        if metadata is None or json.loads(metadata['prompt']) != args[1]:
            checkpoint.reset()
            checkpoint.store('prompt', {'x': torch.ones(1)},
                             {'prompt': json.dumps(args[1])}, priority=2)
        args[4]['prompt_checked'] = True
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
