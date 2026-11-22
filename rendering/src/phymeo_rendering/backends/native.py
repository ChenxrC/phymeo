"""Home for the native graphics backend and its private device layer (RHI).

The backend will translate a native pass plan into resource bindings and commands.
The device layer will own GPU buffers, textures, pipelines, barriers, queue
submission, completion fences and readback. Vulkan is the proposed first device.
Concrete handles and synchronization APIs are deferred until the first device port.
"""
