// Runs simplify.js off the main thread, one job (one material group) at a
// time. This exists because a single CPU pass in simplify.js is fully
// synchronous -- it only yields to the event loop *between* passes, never
// during one -- and for a large group (hundreds of thousands of faces,
// common in real multi-part CAD assemblies) a single pass can take long
// enough to freeze the entire tab for a very long stretch. Running it in a
// module Worker keeps the page responsive regardless of how long any one
// job takes; the caller still sees per-pass progress via postMessage.
//
// WebGPU is also usable from inside a dedicated worker in modern browsers,
// so the GPU path works here too -- import.meta / dynamic import isn't
// needed since this is loaded as `new Worker(url, { type: "module" })`.

import { simplifyAsync, simplifyWebGPU, isWebGPUSupported, requestWebGPUDevice } from "./simplify.js";

let gpuDevicePromise = null;
function getSharedGPUDevice() {
  if (!gpuDevicePromise) gpuDevicePromise = requestWebGPUDevice();
  return gpuDevicePromise;
}

self.onmessage = async (event) => {
  const { id, vertices, faces, targetNumFaces, opts, useGpu } = event.data;

  const onProgress = (info) => {
    self.postMessage({ id, type: "progress", ...info });
  };

  try {
    let result;
    if (useGpu && isWebGPUSupported()) {
      try {
        const device = await getSharedGPUDevice();
        result = await simplifyWebGPU(vertices, faces, targetNumFaces, { ...opts, onProgress, device });
      } catch (err) {
        self.postMessage({ id, type: "log", message: `WebGPU failed in worker (${err.message}) -- falling back to CPU.` });
        result = await simplifyAsync(vertices, faces, targetNumFaces, { ...opts, onProgress });
      }
    } else {
      result = await simplifyAsync(vertices, faces, targetNumFaces, { ...opts, onProgress });
    }

    // Transfer the result buffers back instead of copying them.
    const transfer = [result.vertices.buffer, result.faces.buffer];
    if (result.faceMask) transfer.push(result.faceMask.buffer);
    self.postMessage({ id, type: "done", vertices: result.vertices, faces: result.faces, faceMask: result.faceMask }, transfer);
  } catch (err) {
    self.postMessage({ id, type: "error", message: err.message, stack: err.stack });
  }
};
