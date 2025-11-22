# Client-side image segmentation with SAM2 and MobileSAM
This is a Next.js application that performs image segmentation using Meta's Segment Anything Model V2 (SAM2) and MobileSAM with onnxruntime-web. All the processing is done on the client side.

Demo at [sam2-seven.vercel.app](https://sam2-seven.vercel.app/)

https://github.com/user-attachments/assets/0d3b9f3b-2ab1-4627-9662-fca1a7cc2289

# Features
* Multiple model support with switchable architecture:
  * [Meta's SAM2 Tiny](https://ai.meta.com/blog/segment-anything-2/) - Accurate, reliable segmentation (default)
  * [MobileSAM Tiny](https://github.com/ChaoningZhang/MobileSAM) - Faster inference (~345ms vs ~700ms encode), smaller download (45MB vs 151MB)
* [onnxruntime-web](https://github.com/microsoft/onnxruntime) for model inference
* webgpu-accelerated if GPU available and supported by browser, cpu if not
* Model storage using [OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) ([not working](https://bugs.webkit.org/show_bug.cgi?id=231706) in Safari)
* Image upload or load from URL
* Mask decoding based on point prompt (positive and negative points)
* Mask refinement with iterative clicks
* Cropping
* Tested on macOS with Edge (webgpu, cpu), Chrome (webgpu, cpu), Firefox (cpu only), Safari (cpu only)
* Fails on iOS (17, iPhone SE), not sure why

# Installation
Clone the repository:

```
git clone https://github.com/geronimi73/next-sam
cd next-sam
npm install
npm run dev
```

Open your browser and visit http://localhost:3000 

# Usage
1. Select a model from the dropdown (SAM2 Tiny or MobileSAM Tiny).
2. Upload an image or load from URL.
3. Click the "Encode image" button to start encoding the image.
4. Once the encoding is complete, click on the image to decode masks.
5. Left click to include area ("positive click"), right click to exclude area ("negative click").
6. Continue clicking to refine the mask iteratively.
7. Click the "Crop" button to crop the image using the decoded mask.

# Acknowledgements
* [Meta's Segment Anything Model 2](https://ai.meta.com/blog/segment-anything-2/)
* [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) - Faster, lighter SAM variant
* [@raedle](https://github.com/raedle) for adding [positive/negative clicks](https://github.com/geronimi73/next-sam/pull/1)
* [onnxruntime](https://github.com/microsoft/onnxruntime)
* [Shadcn/ui components](https://ui.shadcn.com/)
* [transformer.js](https://github.com/huggingface/transformers.js)
* https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything
* https://github.com/lucasgelfond/webgpu-sam2
* https://github.com/microsoft/onnxruntime-inference-examples
* https://github.com/ChaoningZhang/MobileSAM
