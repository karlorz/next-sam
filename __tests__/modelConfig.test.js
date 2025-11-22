/**
 * Model Configuration Tests
 *
 * These tests validate model configurations and tensor formats to catch
 * common errors before runtime (like tensor shape mismatches).
 */

import { MODEL_CONFIG, DEFAULT_MODEL } from '../app/modelConfig';

describe('Model Configuration', () => {
  test('DEFAULT_MODEL exists in MODEL_CONFIG', () => {
    expect(MODEL_CONFIG[DEFAULT_MODEL]).toBeDefined();
  });

  test('all models have required fields', () => {
    const requiredFields = [
      'id',
      'name',
      'description',
      'encoderUrl',
      'decoderUrl',
      'imageSize',
      'maskSize',
      'modelType',
      'encoderInputName',
      'useBatchDimension',
      'tensorFormat',
    ];

    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      requiredFields.forEach(field => {
        expect(config[field]).toBeDefined();
      });
    });
  });

  test('all model URLs are valid HTTPS URLs', () => {
    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      expect(config.encoderUrl).toMatch(/^https:\/\/.+/);
      expect(config.decoderUrl).toMatch(/^https:\/\/.+/);
    });
  });

  test('tensorFormat is valid (CHW or HWC)', () => {
    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      expect(['CHW', 'HWC']).toContain(config.tensorFormat);
    });
  });

  test('imageSize and maskSize have correct structure', () => {
    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      expect(config.imageSize).toHaveProperty('w');
      expect(config.imageSize).toHaveProperty('h');
      expect(typeof config.imageSize.w).toBe('number');
      expect(typeof config.imageSize.h).toBe('number');

      expect(config.maskSize).toHaveProperty('w');
      expect(config.maskSize).toHaveProperty('h');
      expect(typeof config.maskSize.w).toBe('number');
      expect(typeof config.maskSize.h).toBe('number');
    });
  });

  test('modelType is valid', () => {
    const validModelTypes = ['mobilesam', 'sam2'];
    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      expect(validModelTypes).toContain(config.modelType);
    });
  });
});

describe('Tensor Shape Validation', () => {
  // Helper to compute expected tensor shape
  function getExpectedTensorShape(config) {
    const { imageSize, tensorFormat, useBatchDimension } = config;
    const h = imageSize.h;
    const w = imageSize.w;
    const c = 3; // RGB channels

    if (tensorFormat === 'HWC') {
      // HWC format: [H, W, C] without batch
      return [h, w, c];
    } else {
      // CHW format: [B, C, H, W] or [C, H, W]
      return useBatchDimension ? [1, c, h, w] : [c, h, w];
    }
  }

  test('MobileSAM expects HWC format [H, W, 3]', () => {
    const mobileSamConfig = MODEL_CONFIG['mobilesam_tiny'];
    if (mobileSamConfig) {
      const shape = getExpectedTensorShape(mobileSamConfig);
      expect(shape).toEqual([1024, 1024, 3]);
      expect(mobileSamConfig.tensorFormat).toBe('HWC');
    }
  });

  test('SAM2 expects CHW format [1, 3, H, W]', () => {
    const sam2Config = MODEL_CONFIG['sam2_tiny'];
    if (sam2Config) {
      const shape = getExpectedTensorShape(sam2Config);
      expect(shape).toEqual([1, 3, 1024, 1024]);
      expect(sam2Config.tensorFormat).toBe('CHW');
    }
  });

  test('HWC models should not use batch dimension', () => {
    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      if (config.tensorFormat === 'HWC') {
        expect(config.useBatchDimension).toBe(false);
      }
    });
  });
});

describe('Model URL Accessibility', () => {
  // This test ensures URLs point to HuggingFace (reliable hosting)
  test('model URLs use HuggingFace hosting', () => {
    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      expect(config.encoderUrl).toMatch(/huggingface\.co/);
      expect(config.decoderUrl).toMatch(/huggingface\.co/);
    });
  });
});

describe('Mask Refinement Handling', () => {
  test('maskSize is defined for decoder input (may differ from actual output)', () => {
    // Note: The decoder may output masks at different resolutions than maskSize
    // MobileSAM outputs 1024x1024, SAM2 outputs 256x256
    // The code stores actual output dimensions (prevMaskDims) for refinement
    Object.entries(MODEL_CONFIG).forEach(([modelId, config]) => {
      expect(config.maskSize).toBeDefined();
      expect(config.maskSize.w).toBeGreaterThan(0);
      expect(config.maskSize.h).toBeGreaterThan(0);
    });
  });

  test('models have consistent imageSize for mask coordinate normalization', () => {
    // All models should use same imageSize for consistent point coordinate handling
    const imageSizes = Object.values(MODEL_CONFIG).map(c => `${c.imageSize.w}x${c.imageSize.h}`);
    const uniqueSizes = [...new Set(imageSizes)];
    // Currently all models use 1024x1024
    expect(uniqueSizes.length).toBe(1);
    expect(uniqueSizes[0]).toBe('1024x1024');
  });
});
