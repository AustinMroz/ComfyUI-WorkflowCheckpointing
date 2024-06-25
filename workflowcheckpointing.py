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
import heapq

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

class FetchQueue:
    """Modified priority queue implementation that tracks inflight and allows priority modification"""
    def __init__(self):
        self.lock = threading.RLock()
        self.queue = []# queue contains priority, url, future
        self.count = 0
        self.consumed = {}
        self.new_items = asyncio.Event()
    def update_priority(self, i, priority):
        #lock must already be acquired
        future = self.queue[i][3]
        if priority < self.queue[i][0]:
            #priority is increased, invalidate old
            self.queue[i] = (self.queue[i][0], self.queue[i][1], None, None)
            heapq.heappush(self.queue, (priority, self.count, item, future))
            self.count += 1
    def enqueue_checked(self, item, priority):
        with self.lock:
            if item in self.consumed:
                #check and update priority on sub chunks
                updated_entries = []
                for i in range(len(self.queue)):
                    if self.queue[i][2].startswith('http'):
                        continue
                    url = self.queue[i][2][self.queue[i][2].find(',')+1:]
                    if url == item:
                        if self.queue[i][0] >= priority -1:
                            break
                        updated_entires.push(priority-1, self.queue[i][2], self.queue[i][3])
                        self.queue[i] = (self.queue[i][0], self.queue[i][1], None, None)
                for e in updated_entries:
                    self.heapq.heappush(self.queue, (e[0], self.count, e[1], e[2]))
                    self.count += 1
                return self.consumed[item]
            for i in range(len(self.queue)):
                if self.queue[i][2] == item:
                    future = self.queue[i][3]
                    self.update_priority(i, priority)
                    return future
            future = asyncio.Future()
            heapq.heappush(self.queue, (priority, self.count, item, future))
            self.count += 1
            self.new_items.set()
            return future
    async def get(self):
        while True:
            await self.new_items.wait()
            with self.lock:
                priority, _, item, future = heapq.heappop(self.queue)
                if len(self.queue) == 0:
                    self.new_items.clear()
                if item is not None:
                    self.consumed[item] = future
                    return priority, item, future

class FetchLoop:
    def __init__(self):
        self.queue = FetchQueue()
        self.semaphore = asyncio.Semaphore(5)
        self.cs = aiohttp.ClientSession()
        event_loop = server.PromptServer.instance.loop
        self.process_loop = event_loop.create_task(self.loop())
        os.makedirs("fetches", exist_ok=True)
    async def loop(self):
        event_loop = server.PromptServer.instance.loop
        while True:
            await self.semaphore.acquire()
            event_loop.create_task(self.fetch(*(await self.queue.get())))
    def enqueue(self, url, priority=0):
        return self.queue.enqueue_checked(url, priority)
    async def fetch(self, priority, url, future):
        chunk_size = 2**26 #64MB
        headers = {}
        if url.startswith(base_url):
            headers.update(await get_header())
        if not url.startswith('http'):
            ci = url.find(',')
            chunk_index = int(url[:ci])
            url = url[ci+1:]
            headers.update({'Range': f'bytes={chunk_index*chunk_size}-{(chunk_index+1)*chunk_size}'})
        else:
            chunk_index = None
        filename = os.path.join('fetches', string_hash(url))
        is_big = url.startswith('https://civitai.com') or url.startswith('https://huggingface.co')
        if not is_big or chunk_index is not None:
            try:
                async with self.cs.get(url, headers=headers) as r:
                    if chunk_index is not None:
                        with open(filename, 'r+b') as f:
                            f.seek(chunk_index)
                            f.write(await r.read())
                    else:
                        with open(filename, 'wb') as f:
                            f.write(await r.read())
                future.set_result(filename)
            except:
                future.set_result(None)
                raise
            finally:
                self.semaphore.release()
            return
        try:
            #get size
            async with self.cs.head(url, allow_redirects=True) as r:
                total_size = int(r.headers['Content-Length'])
        finally:
            self.semaphore.release()
        #clear file
        with open(filename, 'w') as f:
            pass
        chunks = []
        #TODO: Race condition where multiple model downloads initiate before priority drops
        for i in range(0, total_size, chunk_size):
            chunks.append(self.enqueue(f'{i},{url}', priority-1))
        await asyncio.gather(*chunks)
        future.set_result(filename)
fetch_loop = FetchLoop()
async def prepare_file(url, path, priority):
    hashloc = await fetch_loop.enqueue(url, priority)
    if os.path.exists(path):
        os.remove(path)
    #TODO consider if symlinking would be better
    os.link(hashloc, path)

ORGANIZATION = os.environ.get('SALAD_ORGANIZATION', None)
if ORGANIZATION is not None:
    base_url = 'https://storage-api.salad.com/organizations/' + ORGANIZATION +'/files'
class NetCheckpoint:
    def __init__(self):
        self.requestloop = RequestLoop()
        self.has_warned_size = False
        assert ORGANIZATION is not None
    def store(self, unique_id, tensors, metadata, priority=0):
        return
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
def string_hash(s):
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return h.hexdigest()
def fetch_remote_file(url, filepath, file_hash=None):
    assert filepath.find("..") == -1, "Paths may not contain .."
    return prepare_file(url, filepath, -1)


async def fetch_remote_files(remote_files, uid=None):
    #TODO: Add requested support for zip files?
    if uid is not None:
        checkpoint_base = '/'.join([base_url, uid, 'checkpoint'])
        async with self.cs.get(base_url, headers=get_header()) as r:
            js = await r.json()
        files = js['files']
        checkpoints =  list(filter(lambda x: x['url'].startswith(checkpoint_base), files))
        for cp in checkpoints:
            cp['filepath'] = os.path.join('input/checkpoint',
                                          cp['url'][len(checkpoint_base)+1:])
        remote_files = itertools.chain(remote_files, checkpoints)
    fetches = []
    for f in remote_files:
        fetches.append(fetch_remote_file(**f))
    if len(fetches) > 0:
        await asyncio.gather(*fetches)

completion_futures = {}
def add_future(json_data):
    index = max(completion_futures.keys())
    json_data['extra_data']['completion_future'] = index
    return json_data
server.PromptServer.instance.add_on_prompt_handler(add_future)

prompt_route = next(filter(lambda x: x.path  == '/prompt' and x.method == 'POST',
                           server.PromptServer.instance.routes))
original_post_prompt = prompt_route.handler
async def post_prompt_remote(request):
    json_data = await request.json()
    if "SALAD_ORGANIZATION" in os.environ:
        extra_data = json_data.get("extra_data", {})
        #NOTE: Rendered obsolete by existing infrastructure, can be pruned
        remote_files = extra_data.get("remote_files", [])
        uid = None#temporarily disable s4 for testing
        #uid = json_data.get("client_id", 'local')
        checkpoint.uid = uid
        await fetch_remote_files(remote_files, uid=uid)
        if 'prompt' not in json_data:
            return server.web.json_response("PreLoad Complete")
    f = asyncio.Future()
    index = max(completion_futures.keys(),default=0)+1
    completion_futures[index] = f
    base_res = await original_post_prompt(request)
    res = await f
    completion_futures.pop(index)
    res = base_res.text[:-1] + ', "outputs": ' + json.dumps(res) + '}'
    return server.web.Response(body=res)
#Dangerous
object.__setattr__(prompt_route, 'handler', post_prompt_remote)

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
    prev_outputs = None
    if 'checkpoints' in extra_data:
        checkpoint.update(extra_data.pop('checkpoints'))
    if 'prompt_checked' not in args[4]:
        metadata = checkpoint.get('prompt')[1]
        if metadata is None or json.loads(metadata['prompt']) != args[1]:
            checkpoint.reset()
            checkpoint.store('prompt', {'x': torch.ones(1)},
                             {'prompt': json.dumps(args[1])}, priority=2)
        args[4]['prompt_checked'] = True
        prev_outputs = {}
        os.makedirs("temp", exist_ok=True)
        #TODO: Consider subdir recursing?
        for item in itertools.chain(os.scandir("output"), os.scandir("temp")):
            if item.is_file():
                prev_outputs[item.path] = item.stat().st_mtime
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
    if prev_outputs is not None:
        outputs = []
        for item in itertools.chain(os.scandir("output"), os.scandir("temp")):
            if item.is_file() and prev_outputs.get(item.path, 0) < item.stat().st_mtime:
                outputs.append(item.path)
        if 'completion_future' in extra_data:
            completion_futures[extra_data['completion_future']].set_result(outputs)
    return res

comfy.samplers.KSAMPLER = CheckpointSampler
execution.recursive_execute = recursive_execute_injection

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
