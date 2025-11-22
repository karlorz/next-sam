# Project Overview

Next.js application for client-side image segmentation using Meta's SAM2 and MobileSAM with onnxruntime-web.

## Key Technologies
- Next.js, React, onnxruntime-web, Web Workers, Tailwind CSS, shadcn/ui

## Architecture
- `app/page.jsx` - Main UI and user interaction
- `app/SAM2.js` - Model abstraction (download, encode, decode)
- `app/worker.js` - Web worker for inference
- `app/modelConfig.js` - Model configurations
- `lib/imageutils.js` - Image manipulation utilities

## Scripts
- `npm run dev` - Development server
- `npm run build` - Production build
- `npm test` - Run tests

---

# MCP Tools for Documentation

## ALWAYS use these MCP tools for up-to-date information:

### Context7 MCP (`mcp__context7__`)
- **resolve-library-id** - Find library ID before fetching docs
- **get-library-docs** - Fetch current documentation with code examples
- Use for: onnxruntime-web, React, Next.js, any npm package docs

### DeepWiki MCP (`mcp__deepwiki__`)
- **read_wiki_structure** - Get repo documentation topics
- **read_wiki_contents** - View repo documentation
- **ask_question** - Ask questions about any GitHub repository
- Use for: ChaoningZhang/MobileSAM, facebookresearch/segment-anything, model implementations

### When to use:
- Before implementing ONNX model features → Context7 for onnxruntime-web docs
- Debugging model issues → DeepWiki for original repo (MobileSAM, SAM2)
- Checking tensor formats → Both for latest examples
- Verifying preprocessing → DeepWiki for model-specific requirements

---

# Testing Requirements

## Pre-Commit
All changes must pass `npm test` before committing.

## Critical Tests (`__tests__/modelConfig.test.js`)
- Model configs have required fields
- URLs are valid HTTPS
- Tensor formats are valid (CHW/HWC)
- Image/mask sizes correct

## Test Philosophy
- Catch errors at build time, not runtime
- Add tests when bugs are found

---

# ONNX Runtime Web Rules

## 1. Stack Overflow Prevention
NEVER use spread operator on large tensors (1M+ elements). Use manual iteration for min/max.

## 2. Tensor Formats
- SAM2: CHW `[1, 3, H, W]` with batch dimension
- MobileSAM: HWC `[H, W, 3]` without batch dimension

## 3. Input Range (Critical for MobileSAM)
- Acly ONNX export expects **0-255 range** (includes normalization in graph)
- Use `inputRange: 255` in config to scale 0-1 back to 0-255

## 4. Mask IoU Scores
- Scores >1.0 are valid high-confidence (not errors)
- Select maximum score regardless of value

## 5. Mask Refinement
- MobileSAM outputs 1024x1024, expects 256x256 refinement input
- SAM2 outputs 256x256, expects 256x256 refinement input
- Always resize to maskSize for refinement

## 6. Model Config Required Fields
- id, name, encoderUrl, decoderUrl
- imageSize, maskSize, modelType
- encoderInputName, useBatchDimension, tensorFormat
- inputRange (for MobileSAM: 255)

## 7. Worker Lifecycle
- Terminate old worker before creating new one
- Re-initialize with new model config on switch
- Reset encoding state but preserve loaded image

---

# Model Quality

## SAM2 (Default)
- Reliable, accurate segmentation
- IoU range: 0.08-0.13 (valid)
- 256x256 mask output

## MobileSAM (Alternative)
- Faster (~345ms encode vs ~700ms)
- Smaller (45MB vs 151MB)
- Requires inputRange: 255
- 1024x1024 mask output

---

# Known Issues

## MobileSAM Input Range
- **Issue**: Full-image masks when using 0-1 range
- **Fix**: Use inputRange: 255 in config
- **Reference**: MOBILESAM_ANALYSIS.md

## Stack Overflow
- **Issue**: Spread on large arrays
- **Fix**: Manual iteration

## Mask Dimensions
- **Issue**: Output size ≠ refinement input
- **Fix**: Resize to maskSize for refinement

---

# References

- **ONNX Runtime Web**: Use Context7 MCP for latest docs
- **SAM2/MobileSAM**: Use DeepWiki MCP for repo-specific info
- **Model Analysis**: See MOBILESAM_ANALYSIS.md
- **Tests**: See __tests__/modelConfig.test.js
