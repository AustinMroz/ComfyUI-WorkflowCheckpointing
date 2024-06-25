import server
#Check for availability of workflow checkpointing
import importlib
#wcp = importlib.import_module('custom_nodes.ComfyUI-WorkflowCheckpointing.workflowcheckpointing')

web = server.web
ps = server.PromptServer.instance

finished_startup = False
original_server_start = ps.start
async def server_start(*args, **kwargs):
    args = list(args)
    original_on_start= args[3]
    def on_start(scheme, address, port):
        if original_on_start is not None:
            original_on_start(scheme, address, port)
        global finished_startup
        finished_startup= True
    args[3] = on_start
    return await original_server_start(*args, **kwargs)
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
    if len(ps[0]) == 0 and len(ps[1]) == 0:
        return web.json_response(current_queue)
    return web.json_response(current_queue, status=503)
