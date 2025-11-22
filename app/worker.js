import { SAM2 } from "./SAM2";
import { Tensor } from "onnxruntime-web";
import { MODEL_CONFIG, DEFAULT_MODEL } from "./modelConfig.js";

// Lazy initialization - created when ping message received
let sam = null;

const stats = {
  modelId: null,
  device: "unknown",
  downloadModelsTime: [],
  encodeImageTimes: [],
  decodeTimes: [],
};

// Convert CHW [C, H, W] to HWC [H, W, C] for MobileSAM
function chwToHwc(chwArray, channels, height, width) {
  const hwcArray = new Float32Array(chwArray.length);
  const channelSize = height * width;

  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      for (let c = 0; c < channels; c++) {
        const chwIdx = c * channelSize + h * width + w;
        const hwcIdx = h * width * channels + w * channels + c;
        hwcArray[hwcIdx] = chwArray[chwIdx];
      }
    }
  }

  return hwcArray;
}

// Apply ImageNet normalization: (pixel * scale - mean) / std
function applyNormalization(data, channels, height, width, normalization) {
  const { mean, std, scale } = normalization;
  const channelSize = height * width;
  const normalizedData = new Float32Array(data.length);

  for (let c = 0; c < channels; c++) {
    for (let i = 0; i < channelSize; i++) {
      const idx = c * channelSize + i;
      // Input is 0-1, scale to 0-255, then normalize
      normalizedData[idx] = (data[idx] * scale - mean[c]) / std[c];
    }
  }

  return normalizedData;
}

self.onmessage = async (e) => {
  // console.log("worker received message")

  const { type, data } = e.data;

  if (type === "ping") {
    // Receive modelId from main thread, default to DEFAULT_MODEL
    const modelId = data?.modelId || DEFAULT_MODEL;
    const modelConfig = MODEL_CONFIG[modelId];

    // Create SAM2 instance with selected model config
    sam = new SAM2(modelConfig);
    stats.modelId = modelId;

    self.postMessage({ type: "downloadInProgress" });
    const startTime = performance.now();
    await sam.downloadModels();
    const durationMs = performance.now() - startTime;
    stats.downloadModelsTime.push(durationMs);

    self.postMessage({ type: "loadingInProgress" });
    const report = await sam.createSessions();

    stats.device = report.device;

    self.postMessage({ type: "pong", data: report });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "encodeImage") {
    const { float32Array, shape } = data;

    // Handle tensor format conversion based on model config
    let tensorData = float32Array;
    let tensorShape = shape;

    // Apply normalization if model requires it (e.g., MobileSAM needs ImageNet normalization)
    if (sam.modelConfig.normalization) {
      const [, channels, height, width] = shape; // shape is always [1, 3, H, W] from page.jsx
      tensorData = applyNormalization(tensorData, channels, height, width, sam.modelConfig.normalization);
    }

    // Scale input range if needed (e.g., MobileSAM expects 0-255)
    if (sam.modelConfig.inputRange) {
      const scaledData = new Float32Array(tensorData.length);
      for (let i = 0; i < tensorData.length; i++) {
        scaledData[i] = tensorData[i] * sam.modelConfig.inputRange;
      }
      tensorData = scaledData;
    }

    if (sam.modelConfig.tensorFormat === "HWC") {
      // Convert CHW to HWC for MobileSAM
      // shape from page.jsx is always [1, 3, H, W]
      // Strip batch dimension if model doesn't use it
      const actualShape = sam.modelConfig.useBatchDimension ? shape : shape.slice(1);
      const [channels, height, width] = actualShape;

      tensorData = chwToHwc(tensorData, channels, height, width);
      // MobileSAM expects [H, W, C] without batch dimension
      tensorShape = [height, width, channels];
    } else {
      // CHW format for SAM2 - use as is (shape already correct)
      tensorShape = sam.modelConfig.useBatchDimension ? shape : shape.slice(1);
    }

    const imgTensor = new Tensor("float32", tensorData, tensorShape);

    const startTime = performance.now();
    await sam.encodeImage(imgTensor);
    const durationMs = performance.now() - startTime;
    stats.encodeImageTimes.push(durationMs);

    self.postMessage({
      type: "encodeImageDone",
      data: { durationMs: durationMs },
    });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "decodeMask") {
    const {points, maskArray, maskShape} = data;

    const startTime = performance.now();

    let decodingResults;
    if (maskArray) {
      const maskTensor = new Tensor("float32", maskArray, maskShape);
      decodingResults = await sam.decode(points, maskTensor);
    } else {
      decodingResults = await sam.decode(points);
    }
    // decodingResults = Tensor [B=1, Masks, W, H]

    const durationMs = performance.now() - startTime;
    stats.decodeTimes.push(durationMs);

    self.postMessage({ type: "decodeMaskResult", data: decodingResults });
    self.postMessage({ type: "stats", data: stats });
  } else if (type === "stats") {
    self.postMessage({ type: "stats", data: stats });
  } else {
    throw new Error(`Unknown message type: ${type}`);
  }
};
