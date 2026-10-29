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
export const MIRROR_HOST = "https://hf-mirror.com";

/** 在指定 host 上创建 pipeline；失败时抛出，由调用方决定是否换源 */
async function buildPipeline({ dev, dtype, onProgress }) {
  const dl = (info) => {
    if (!onProgress) return;
    if (info?.status === "progress" && info.file) {
      const pct = Number.isFinite(info.progress) ? info.progress : 0;
      onProgress({ phase: "download", file: info.file, pct });
    }
  };
  return pipeline("feature-extraction", MODEL_ID, { device: dev, dtype, progress_callback: dl });
}

export async function createEmbedder({ dtype = "fp16", device, onProgress } = {}) {
  const preferred = device || detectBackend();
  const attempts = preferred === "webgpu" ? ["webgpu", "wasm"] : ["wasm"];
  let extractor = null;
  let usedBackend = null;
  let usedDtype = dtype;
  let lastErr = null;

  for (const dev of attempts) {
    for (const host of [null, MIRROR_HOST]) {
      try {
        if (host) env.remoteHost = host;
        extractor = await buildPipeline({ dev, dtype, onProgress });
        usedBackend = dev;
        if (host) console.info("[embedder] 主站不可用，已切换到镜像", host);
        break;
      } catch (err) {
        lastErr = err;
        // fp16 在部分 WASM 后端不可用，退回 fp32 再试一次
        if (dtype !== "fp32" && dev === "wasm") {
          try {
            if (host) env.remoteHost = host;
            extractor = await buildPipeline({ dev, dtype: "fp32", onProgress });
            usedBackend = dev;
            usedDtype = "fp32";
            break;
          } catch (e2) {
            lastErr = e2;
          }
        }
      }
    }
    if (extractor) break;
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
