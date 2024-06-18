# ComfyUI-WorkflowCheckpointing
Automatically creates checkpoints during workflow execution. If If an workflow is canceled or ComfyUI crashes mid-execution, then these checkpoints are used when the workflow is re-queued to resume execution with minimal progress loss.

## Configuration
By default, checkpoints are saved localy to a checkpoint folder.

Networked checkpoints backed by the Salad Simple Storage Service can be enabled by setting the `SALAD_ORGANIZATION` environment variable. When deployed to salad, authentication is automatically pulled, but an api key can be applied when testing locally by setting `SALAD_API_KEY`

If a request sent to `/prompt` contains a `client_id`. It is utilized to allow independent caching of executions by multiple users.

To facilitate testing `FORCE_CRASH_AT` can be set to an integer to terminate workflow execution at a given sampling step
