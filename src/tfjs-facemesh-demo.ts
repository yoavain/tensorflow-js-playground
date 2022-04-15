import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import type { Face, FaceLandmarksDetector, MediaPipeFaceMeshTfjsModelConfig } from "@tensorflow-models/face-landmarks-detection";
import { tensor3d, Tensor3D } from "@tensorflow/tfjs-node-gpu";
import { decodeImage, encodeImage } from "./tfjs-image-utils";

type Shape = [number, number, number];

const setAllLayers = (uint8Array: Uint8Array, x: number, y: number, shape: Shape) => {
    const [width, , channels] = shape;
    for (let channel = 0; channel < channels; channel++) {
        uint8Array[channel * channels * (x + y * width)] = 0;
    }
}

const setWide = (uint8Array: Uint8Array, floatX: number, floatY: number, shape: Shape) => {
    const x = Math.round(floatX);
    const y = Math.round(floatY);
    setAllLayers(uint8Array, x, y, shape);

    // Padding + 1
    const [maxX, maxY] = shape;
    const xMin1 = Math.max(0, x - 1);
    const xPlus1 = Math.min(maxX, x + 1);
    const yMin1 = Math.max(0, y - 1);
    const yPlus1 = Math.min(maxY, y + 1);
    setAllLayers(uint8Array, xMin1, yMin1, shape);
    setAllLayers(uint8Array, xMin1, y, shape);
    setAllLayers(uint8Array, xMin1, yPlus1, shape);
    setAllLayers(uint8Array, x, yMin1, shape);
    setAllLayers(uint8Array, x, yPlus1, shape);
    setAllLayers(uint8Array, xPlus1, yMin1, shape);
    setAllLayers(uint8Array, xPlus1, y, shape);
    setAllLayers(uint8Array, xPlus1, yPlus1, shape);
};

const main = async (imagePath: string) => {
    // Load the MediaPipe facemesh model.
    const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
    const detectorConfig: MediaPipeFaceMeshTfjsModelConfig = {
        runtime: "tfjs",
        refineLandmarks: false
    }
    const detector: FaceLandmarksDetector = await faceLandmarksDetection.createDetector(model, detectorConfig);

    const input: Tensor3D = await decodeImage(imagePath);
    let outputData: Uint8Array = new Uint8Array(input.dataSync());

    const predictions: Face[] = await detector.estimateFaces(input);
    let inputShape: Shape = input.shape;

    if (predictions.length > 0) {
        for (let i = 0; i < predictions.length; i++) {
            const keypoints = predictions[i].keypoints;

            // Log facial keypoints.
            for (let i = 0; i < keypoints.length; i++) {
                const { x, y, z } = keypoints[i];

                setWide(outputData, x, y, inputShape);
                console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
            }
        }
    }

    let tensor: Tensor3D = tensor3d(outputData, inputShape);

    await encodeImage(tensor, imagePath, "facemesh");
};

// run
if (process.argv.length < 3) {
    console.log('please pass an image to process. ex:');
    console.log('   ts-node tfjs-facemesh-demo.ts /path/to/image.jpg');
}
else {
    let imagePath = process.argv[2];
    main(imagePath).then(() => {})
}