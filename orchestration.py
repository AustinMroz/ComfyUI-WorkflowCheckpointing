import json
import os
import server
#Check for availability of workflow checkpointing
import aiohttp
from .workflowcheckpointing import post_prompt_remote

web = server.web
ps = server.PromptServer.instance

finished_startup = False
original_server_start = ps.start
async def server_start(address, port, verbose=True, call_on_start=None):
    original_on_start= call_on_start
    def on_start(*args, **kwargs):
        if original_on_start is not None:
            original_on_start(*args, **kwargs)
        global finished_startup
        finished_startup= True
    return await original_server_start(address, port, verbose, on_start)
ps.start = server_start


@ps.routes.get("/health")
async def heath(request):
    #while any of the server endpoints could likely be used
    return web.json_response([])

@ps.routes.get("/startup")
async def startup(request):
    if finished_startup:
        return web.json_response([])
    return web.Response(status=503)

@ps.routes.get("/ready")
async def ready(request):
    current_queue = ps.prompt_queue.get_current_queue()
    if len(current_queue[0]) == 0 and len(current_queue[1]) == 0:
        return web.json_response(current_queue)
    return web.json_response(current_queue, status=503)

async def websocket_loop():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(os.environ["ORCHESTRATION_SERVER"]) as ws:
            print("connected to server")
            async for msg in ws:
                print("got command")
                if msg.type == aiohttp.WSMsgType.TEXT:
                    breakpoint()
                    js = msg.json()
                    match js['command']:
                        case 'prompt':
                            #wrap as mock request
                            class MockRequest:
                                async def json(self):
                                    return js['data']
                            resp = await post_prompt_remote(MockRequest())
                            resp = json.loads(resp.body._value)
                        case "queue":
                            resp = ps.prompt_queue.get_current_queue()
                        case "files":
                            #Return a list of files, not yet implemented
                            resp = "Not yet implemented"
                        case _:
                            resp = "Unknown command"
                    print(resp)
                    await ws.send_json(resp)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    await ws.send_json("Error")

process_loop = ps.loop.create_task(websocket_loop())
