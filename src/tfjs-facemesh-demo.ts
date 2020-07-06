import type { FaceMesh } from "@tensorflow-models/facemesh";
import * as facemesh from "@tensorflow-models/facemesh";
import * as tf from "@tensorflow/tfjs-node-gpu";
import { Rank, Tensor3D, TensorBuffer } from "@tensorflow/tfjs-node-gpu";
import { decodeImage, encodeImage } from "./tfjs-image-utils";

const setAllLayers = (tensorBuffer: TensorBuffer<Rank, "int32">, x: number, y: number, value: number) => {
    tensorBuffer.set(x, y, 0, value);
    tensorBuffer.set(x, y, 1, value);
    tensorBuffer.set(x, y, 2, value);
}

const setWide = (tensorBuffer: TensorBuffer<Rank, "int32">, x: number, y: number, shape: [number, number, number]) => {
    const roundX = Math.round(x);
    const roundY = Math.round(y);
    const [maxX, maxY] = shape;
    setAllLayers(tensorBuffer, roundX, roundY, 0);
    if (roundX - 1 >= 0) {
        setAllLayers(tensorBuffer, roundX - 1, roundY, 0);
        if (roundY - 1 >= 0) {
            setAllLayers(tensorBuffer, roundX - 1, roundY -1, 0);
        }
        if (roundY + 1 < maxY) {
            setAllLayers(tensorBuffer, roundX + 1, roundY + 1, 0);
        }
    }
    if (roundX + 1 < maxX) {
        setAllLayers(tensorBuffer, roundX + 1, roundY, 0);
        if (roundY - 1 >= 0) {
            setAllLayers(tensorBuffer, roundX - 1, roundY -1, 0);
        }
        if (roundY + 1 < maxY) {
            setAllLayers(tensorBuffer, roundX + 1, roundY + 1, 0);
        }
    }
};

const main = async (imagePath: string) => {
    // Load the MediaPipe facemesh model.
    const model: FaceMesh = await facemesh.load();

    const input: Tensor3D = await decodeImage(imagePath);
    const inputData: Int32Array = await input.data() as Int32Array;

    const predictions = await model.estimateFaces(input);
    let inputShape: [number, number, number] = input.shape;
    let tensorBuffer: TensorBuffer<Rank, "int32"> = tf.buffer(inputShape, "int32", inputData);

    if (predictions.length > 0) {
        for (let i = 0; i < predictions.length; i++) {
            const keypoints: [number, number, number][] = predictions[i].scaledMesh as [number, number, number][];

            // Log facial keypoints.
            for (let i = 0; i < keypoints.length; i++) {
                const [x, y, z] = keypoints[i];

                setWide(tensorBuffer, x, y, inputShape);
                console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
            }
        }
    }

    let tensor: Tensor3D = tensorBuffer.toTensor() as Tensor3D;

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