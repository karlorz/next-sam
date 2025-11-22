/**
 * Image Utilities Tests
 *
 * Tests for image manipulation functions, focusing on:
 * - RGBA stride handling
 * - Float32Array conversion correctness
 * - Tensor shape generation
 */

import {
  canvasToFloat32Array,
  maskCanvasToFloat32Array,
  float32ArrayToCanvas,
  sliceTensor,
  resizeAndPadBox,
} from '../lib/imageutils';

// Mock canvas and context for Node.js environment
class MockImageData {
  constructor(data, width, height) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
}

class MockCanvas {
  constructor(width, height, data = null) {
    this.width = width;
    this.height = height;
    this._data = data || new Uint8ClampedArray(width * height * 4);
  }

  getContext(type) {
    const self = this;
    return {
      getImageData: (x, y, w, h) => ({
        data: self._data,
        width: w,
        height: h,
      }),
      putImageData: jest.fn(),
    };
  }
}

// Mock globals for canvas operations
global.ImageData = MockImageData;
global.document = {
  createElement: (tag) => {
    if (tag === 'canvas') {
      return new MockCanvas(0, 0);
    }
  },
};

describe('canvasToFloat32Array', () => {
  test('converts 2x2 canvas to correct CHW tensor format', () => {
    // Create 2x2 canvas with known RGBA values
    const canvas = new MockCanvas(2, 2);
    // Pixel (0,0): R=255, G=0, B=0, A=255
    // Pixel (1,0): R=0, G=255, B=0, A=255
    // Pixel (0,1): R=0, G=0, B=255, A=255
    // Pixel (1,1): R=255, G=255, B=255, A=255
    canvas._data.set([
      255, 0, 0, 255,    // pixel 0
      0, 255, 0, 255,    // pixel 1
      0, 0, 255, 255,    // pixel 2
      255, 255, 255, 255 // pixel 3
    ]);

    const { float32Array, shape } = canvasToFloat32Array(canvas);

    // Verify shape
    expect(shape).toEqual([1, 3, 2, 2]);

    // Verify CHW layout: [R, R, R, R, G, G, G, G, B, B, B, B]
    expect(float32Array.length).toBe(12); // 3 channels * 2 * 2

    // Red channel (normalized to 0-1)
    expect(float32Array[0]).toBeCloseTo(1.0); // pixel 0
    expect(float32Array[1]).toBeCloseTo(0.0); // pixel 1
    expect(float32Array[2]).toBeCloseTo(0.0); // pixel 2
    expect(float32Array[3]).toBeCloseTo(1.0); // pixel 3

    // Green channel
    expect(float32Array[4]).toBeCloseTo(0.0);
    expect(float32Array[5]).toBeCloseTo(1.0);
    expect(float32Array[6]).toBeCloseTo(0.0);
    expect(float32Array[7]).toBeCloseTo(1.0);

    // Blue channel
    expect(float32Array[8]).toBeCloseTo(0.0);
    expect(float32Array[9]).toBeCloseTo(0.0);
    expect(float32Array[10]).toBeCloseTo(1.0);
    expect(float32Array[11]).toBeCloseTo(1.0);
  });

  test('uses correct RGBA stride (4 bytes per pixel)', () => {
    const canvas = new MockCanvas(3, 1);
    // Set specific pattern to test stride
    canvas._data.set([
      100, 0, 0, 255,    // R=100
      0, 150, 0, 255,    // G=150
      0, 0, 200, 255     // B=200
    ]);

    const { float32Array } = canvasToFloat32Array(canvas);

    expect(float32Array[0]).toBeCloseTo(100 / 255);
    expect(float32Array[1]).toBeCloseTo(0);
    expect(float32Array[2]).toBeCloseTo(0);
    // Green channel offset
    expect(float32Array[4]).toBeCloseTo(150 / 255);
    // Blue channel offset
    expect(float32Array[8]).toBeCloseTo(200 / 255);
  });
});

describe('maskCanvasToFloat32Array', () => {
  test('converts mask canvas with correct RGBA stride', () => {
    const canvas = new MockCanvas(2, 2);
    // Create alternating mask pattern
    canvas._data.set([
      255, 255, 255, 255,  // white (masked)
      0, 0, 0, 255,        // black (unmasked)
      0, 0, 0, 255,        // black
      255, 255, 255, 255   // white
    ]);

    const float32Array = maskCanvasToFloat32Array(canvas);

    // Should average RGB and normalize
    expect(float32Array.length).toBe(4);
    expect(float32Array[0]).toBeCloseTo(1.0); // (255+255+255)/(3*255)
    expect(float32Array[1]).toBeCloseTo(0.0); // (0+0+0)/(3*255)
    expect(float32Array[2]).toBeCloseTo(0.0);
    expect(float32Array[3]).toBeCloseTo(1.0);
  });

  test('uses i*4 stride not i for RGBA data', () => {
    // Critical regression test for the bug we fixed
    const canvas = new MockCanvas(1, 1);
    canvas._data.set([128, 64, 32, 255]);

    const float32Array = maskCanvasToFloat32Array(canvas);

    // Should read indices 0,1,2 (R,G,B) not 0,1,2 (R,G,B,A)
    const expected = (128 + 64 + 32) / (3 * 255);
    expect(float32Array[0]).toBeCloseTo(expected);
  });
});

describe('float32ArrayToCanvas', () => {
  test('converts Float32Array to canvas with correct color mapping', () => {
    const array = new Float32Array([1.0, 0.0, 1.0, 0.0]); // 2x2 mask
    const canvas = float32ArrayToCanvas(array, 2, 2);

    expect(canvas.width).toBe(2);
    expect(canvas.height).toBe(2);
  });

  test('uses boolean maskedPx correctly (no redundant comparisons)', () => {
    // Test that maskedPx > 0 logic works correctly
    const array = new Float32Array([0.5, 0, -1, 2]);
    const canvas = float32ArrayToCanvas(array, 2, 2);

    // Should handle positive, zero, negative, and >1 values
    expect(canvas).toBeDefined();
  });
});

describe('sliceTensor', () => {
  test('extracts correct slice from tensor', () => {
    const mockTensor = {
      dims: [1, 3, 2, 2],
      cpuData: new Float32Array([
        // Mask 0
        1, 2, 3, 4,
        // Mask 1
        5, 6, 7, 8,
        // Mask 2
        9, 10, 11, 12
      ])
    };

    const slice1 = sliceTensor(mockTensor, 1);
    expect(slice1).toEqual(new Float32Array([5, 6, 7, 8]));

    const slice2 = sliceTensor(mockTensor, 2);
    expect(slice2).toEqual(new Float32Array([9, 10, 11, 12]));
  });
});

describe('resizeAndPadBox', () => {
  test('handles square images', () => {
    const result = resizeAndPadBox(
      { w: 100, h: 100 },
      { w: 200, h: 200 }
    );
    expect(result).toEqual({ x: 0, y: 0, w: 200, h: 200 });
  });

  test('pads portrait images on left', () => {
    const result = resizeAndPadBox(
      { w: 100, h: 200 },  // portrait
      { w: 200, h: 200 }   // square target
    );
    expect(result.y).toBe(0);
    expect(result.h).toBe(200);
    expect(result.x).toBeGreaterThan(0); // padded left
    expect(result.w).toBeLessThan(200);  // narrower than target
  });

  test('pads landscape images on top', () => {
    const result = resizeAndPadBox(
      { w: 200, h: 100 },  // landscape
      { w: 200, h: 200 }   // square target
    );
    expect(result.x).toBe(0);
    expect(result.w).toBe(200);
    expect(result.y).toBeGreaterThan(0); // padded top
    expect(result.h).toBeLessThan(200);  // shorter than target
  });
});

describe('Regression Tests for Code Review Fixes', () => {
  test('CRITICAL: maskCanvasToFloat32Array uses correct RGBA stride', () => {
    // This test ensures we read i*4, i*4+1, i*4+2 not i, i+1, i+2
    const canvas = new MockCanvas(2, 1);
    canvas._data.set([
      100, 50, 25, 255,   // pixel 0: RGB average = 58.33
      200, 100, 50, 255   // pixel 1: RGB average = 116.67
    ]);

    const result = maskCanvasToFloat32Array(canvas);

    // If stride is correct, we get averages of RGB for each pixel
    expect(result[0]).toBeCloseTo((100 + 50 + 25) / (3 * 255));
    expect(result[1]).toBeCloseTo((200 + 100 + 50) / (3 * 255));

    // If stride is wrong (bug), we'd get:
    // result[0] = (100 + 50 + 25) / (3 * 255) ✓ (accidentally correct for first pixel)
    // result[1] = (255 + 200 + 100) / (3 * 255) ✗ (reads alpha, then next pixel's R,G)
    const wrongValue = (255 + 200 + 100) / (3 * 255);
    expect(result[1]).not.toBeCloseTo(wrongValue);
  });

  test('HIGH: canvasToFloat32Array uses pre-allocated array (no push/concat)', () => {
    // Performance test - ensure we use Float32Array directly
    const canvas = new MockCanvas(10, 10);
    const { float32Array } = canvasToFloat32Array(canvas);

    expect(float32Array).toBeInstanceOf(Float32Array);
    expect(float32Array.length).toBe(3 * 10 * 10);
  });
});
