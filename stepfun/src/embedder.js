import { pipeline, env } from "@huggingface/transformers";

env.allowLocalModels = false;

export const MODEL_ID = "Xenova/paraphrase-multilingual-MiniLM-L12-v2";

export const DTYPE_FILE = {
  fp32: "onnx/model.onnx",
  fp16: "onnx/model_fp16.onnx",
  q8: "onnx/model_quantized.onnx",
};

export function detectBackend() {
  const hasGpu = typeof navigator !== "undefined" && "gpu" in navigator && !!navigator.gpu;
  return hasGpu ? "webgpu" : "wasm";
}

export function readUrlOptions() {
  const p = new URLSearchParams(location.search);
  const dtype = ["fp32", "fp16", "q8"].includes(p.get("dtype")) ? p.get("dtype") : "fp16";
  const backend = ["webgpu", "wasm"].includes(p.get("backend")) ? p.get("backend") : null;
  return { dtype, backend };
}

/** 模型文件体积（字节），从 HuggingFace 的 Content-Length 读取；失败返回 null */
export async function measureModelBytes(dtype) {
  const file = DTYPE_FILE[dtype];
  if (!file) return null;
  try {
    const url = `https://huggingface.co/${MODEL_ID}/resolve/main/${file}`;
    const head = await fetch(url, { method: "HEAD" });
    const len = Number(head.headers.get("content-length"));
    return Number.isFinite(len) && len > 0 ? len : null;
  } catch {
    return null;
  }
}

/**
 * 端侧 embedding。device 缺省时自动检测（webgpu 优先），失败降级 wasm。
 * 返回实际生效的 devicce 与 dtype，便于界面如实展示。
 */
export async function createEmbedder({ dtype = "fp16", device, onProgress } = {}) {
  const preferred = device || detectBackend();
  const attempts = preferred === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  let extractor = null;
  let usedBackend = null;
  let usedDtype = dtype;
  let lastErr = null;

  const dl = (info) => {
    if (!onProgress) return;
    if (info?.status === "progress" && info.file) {
      const pct = Number.isFinite(info.progress) ? info.progress : 0;
      onProgress({ phase: "download", file: info.file, pct });
    }
  };

  for (const dev of attempts) {
    try {
      extractor = await pipeline("feature-extraction", MODEL_ID, { device: dev, dtype, progress_callback: dl });
      usedBackend = dev;
      break;
    } catch (err) {
      lastErr = err;
      // fp16 在部分 WASM 后端不可用，退回 fp32 再试一次
      if (dtype !== "fp32" && dev === "wasm") {
        try {
          extractor = await pipeline("feature-extraction", MODEL_ID, { device: dev, dtype: "fp32", progress_callback: dl });
          usedBackend = dev;
          usedDtype = "fp32";
          break;
        } catch (e2) {
          lastErr = e2;
        }
      }
    }
  }
  if (!extractor) throw lastErr || new Error("模型加载失败");

  async function embed(texts) {
    const out = await extractor(texts, { pooling: "mean", normalize: true });
    return out.tolist();
  }

  async function dispose() {
    try {
      await extractor.dispose?.();
    } catch {}
  }

  return { device: usedBackend, dtype: usedDtype, requested: { device: preferred, dtype }, model: MODEL_ID, embed, dispose };
}
