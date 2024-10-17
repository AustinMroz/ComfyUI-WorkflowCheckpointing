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
        assert 'SALAD_MACHINE_ID' in os.environ, "SALAD_API_KEY must be provided if not deployed"
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
        checkpoint_base = '/'.join([base_url, uid, 'checkpoint'])
        async with s.get(base_url_path, headers=await get_header()) as r:
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
            try:
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
            except:
                #Exceptions from event loop get swallowed and kill the loop
                import traceback
                traceback.print_exc()
                raise
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
    def requeue(self, future, item, dec_priority=1):
        with self.lock:
            priority = self.consumed[item][1] - dec_priority
            heapq.heappush(self.queue, (priority, self.count, future, None))
            self.count += 1
            self.new_items.set()
    def enqueue_checked(self, item, priority):
        with self.lock:
            if item in self.consumed:
                #TODO: Also update in queue
                #TODO: if complete check etag?
                self.consumed[item][1] = min(self.consumed[item][1], priority)
                return self.consumed[item][0]
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
                    if isinstance(item, str):
                        self.consumed[item] = [future, priority]
                        return priority, item, future
                    else:
                        #item is future
                        item.set_result(True)

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
    def reset(self, url):
        with self.queue.lock:
            if url in self.queue.consumed:
                self.queue.consumed.pop(url)
        hashloc = os.path.join('fetches', string_hash(url))
        if os.path.exists(hashloc):
            os.remove(hashloc)
    def enqueue(self, url, priority=0):
        return self.queue.enqueue_checked(url, priority)
    async def fetch(self, priority, url, future):
        chunk_size = 2**25 #32MB
        headers = {}
        if url.startswith(base_url):
            headers.update(await get_header())
        filename = os.path.join('fetches', string_hash(url))
        try:
            async with self.cs.get(url, headers=headers) as r:
                with open(filename, 'wb') as f:
                    async for chunk in r.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        if not r.content.is_eof():
                            awaken = asyncio.Future()
                            self.queue.requeue(awaken, url)
                            await awaken
            future.set_result(filename)
        except:
            future.set_result(None)
            raise
        finally:
            self.semaphore.release()
        return
fetch_loop = FetchLoop()
async def prepare_file(url, path, priority):
    hashloc = os.path.join('fetches', string_hash(url))
    if not os.path.exists(hashloc):
        hashloc = await fetch_loop.enqueue(url, priority)
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    #TODO consider if symlinking would be better
    os.link(hashloc, path)

ORGANIZATION = os.environ.get('SALAD_ORGANIZATION', None)
if ORGANIZATION is not None:
    base_url_path = '/organizations/' + ORGANIZATION +'/files'
    base_url = 'https://storage-api.salad.com' + base_url_path
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
                fetch_loop.reset('/'.join([base_url, self.uid, 'checkpoint', f'{unique_id}.checkpoint']))
            return
        os.makedirs("input/checkpoint", exist_ok=True)
        for file in os.listdir("input/checkpoint"):
            os.remove(os.path.join("input/checkpoint", file))
            fetch_loop.reset('/'.join([base_url, self.uid, 'checkpoint', file]))

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
        checkpoint_base = '/'.join([base_url_path, uid, 'checkpoint'])
        checkpoint_base = 'https://storage-api.salad.com'+ checkpoint_base
        async with fetch_loop.cs.get(base_url, headers=await get_header()) as r:
            js = await r.json()
        files = js['files']
        checkpoints =  list(filter(lambda x: x['url'].startswith(checkpoint_base), files))
        for cp in checkpoints:
            cp['filepath'] = os.path.join('input/checkpoint',
                                          cp['url'][len(checkpoint_base)+1:])
        remote_files = itertools.chain(remote_files, checkpoints)
    fetches = []
    for f in remote_files:
        fetches.append(fetch_remote_file(f['url'],f['filepath'], f.get('file_hash', None)))
    if len(fetches) > 0:
        await asyncio.gather(*fetches)

prompt_route = next(filter(lambda x: x.path  == '/prompt' and x.method == 'POST',
                           server.PromptServer.instance.routes))
original_post_prompt = prompt_route.handler
async def post_prompt_remote(request):
    if 'dump_req' in os.environ:
        with open('resp-dump.txt', 'wb') as f:
            f.write(await request.read())
        import sys
        sys.exit()
    json_data = await request.json()
    if "SALAD_ORGANIZATION" in os.environ:
        extra_data = json_data.get("extra_data", {})
        remote_files = extra_data.get("remote_files", [])
        uid = json_data.get("client_id", 'local')
        checkpoint.uid = uid
        await fetch_remote_files(remote_files, uid=uid)
    return await original_post_prompt(request)
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

original_recursive_execute = execution.execute
def recursive_execute_injection(*args):
    unique_id = args[3]
    class_type = args[1].get_node(unique_id)['class_type']
    extra_data = args[4]
    if  class_type in SAMPLER_NODES:
        data, metadata = checkpoint.get(unique_id)
        if metadata is not None and 'step' in metadata:
            args[1].get_node(unique_id)['inputs']['latent_image'] = ['checkpointed'+unique_id, 0]
            args[2].outputs.set('checkpointed'+unique_id, [[{'samples': data['x']}]])
        elif metadata is not None and 'completed' in metadata:
            outputs = json.loads(metadata['completed'])
            for x in range(len(outputs)):
                if outputs[x] == 'tensor':
                    outputs[x] = list(data[str(x)])
                elif outputs[x] == 'latent':
                    outputs[x] = [{'samples': l} for l in data[str(x)]]
            args[2].outputs.set(unique_id, outputs)
            return True, None, None

    res = original_recursive_execute(*args)
    #Conditionally save node output
    #TODO: determine which non-sampler nodes are worth saving
    if class_type in SAMPLER_NODES and args[2].outputs.get(unique_id) is not None:
        data = {}
        outputs = args[2].outputs.get(unique_id).copy()
        for x in range(len(outputs)):
            if isinstance(outputs[x][0], torch.Tensor):
                data[str(x)] = torch.stack(outputs[x])
                outputs[x] = 'tensor'
            elif isinstance(outputs[x][0], dict):
                data[str(x)] = torch.stack([l['samples'] for l in outputs[x]])
                outputs[x] = 'latent'
        checkpoint.store(unique_id, data, {'completed': json.dumps(outputs)}, priority=1)
    return res
original_execute = execution.PromptExecutor.execute
def execute_injection(*args, **kwargs):
    metadata = checkpoint.get('prompt')[1]
    if metadata is None or json.loads(metadata['prompt']) != args[1]:
        checkpoint.reset()
        checkpoint.store('prompt', {'x': torch.ones(1)},
                         {'prompt': json.dumps(args[1])}, priority=2)
    original_execute(*args, **kwargs)

comfy.samplers.KSAMPLER = CheckpointSampler
execution.execute = recursive_execute_injection
execution.PromptExecutor.execute = execute_injection
