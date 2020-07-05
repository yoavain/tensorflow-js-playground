import type { FaceMesh } from "@tensorflow-models/facemesh";
import * as facemesh from "@tensorflow-models/facemesh";
import * as tf from "@tensorflow/tfjs-node-gpu";
import { Tensor3D } from "@tensorflow/tfjs-node-gpu";
import { promises as fs } from "fs";
import path from "path";
import { encodePng } from "@tensorflow/tfjs-node-gpu/dist/image";

// convert image to Tensor
const processInput = async (imagePath: string): Promise<Tensor3D> => {
    console.log(`preprocessing image ${imagePath}`);
    const image: Buffer = await fs.readFile(imagePath);
    return tf.node.decodeImage(new Uint8Array(image), 3) as Tensor3D;
}

const annotateImage = async (output: Tensor3D, imagePath) => {
    console.log(`annotating prediction result(s)`);
    try {
        const uint8Array = await encodePng(output);
        const f = path.join(__dirname, "../output", `${path.parse(imagePath).name}-facemesh.png`);
        await fs.writeFile(f, uint8Array);
        console.log(`annotated image saved as ${f}\r\n`);
    }
    catch (err) {
        console.error(err);
    }
}

const main = async (imagePath: string) => {
    // Load the MediaPipe facemesh model.
    const model: FaceMesh = await facemesh.load();

    const input: Tensor3D = await processInput(imagePath);
    const inputData: Int32Array = await input.data() as Int32Array;

    const predictions = await model.estimateFaces(input);
    let tensorBuffer = tf.buffer(input.shape, "int32", inputData);

    if (predictions.length > 0) {
        for (let i = 0; i < predictions.length; i++) {
            const keypoints: [number, number, number][] = predictions[i].scaledMesh as [number, number, number][];

            // Log facial keypoints.
            for (let i = 0; i < keypoints.length; i++) {
                const [x, y, z] = keypoints[i];
                try {
                    tensorBuffer.set(Math.round(x), Math.round(y), 0, 0);
                    tensorBuffer.set(Math.round(x), Math.round(y), 1, 0);
                    tensorBuffer.set(Math.round(x), Math.round(y), 2, 0);
                } catch (e) {
                    console.log(e);
                }

                console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
            }
        }
    }

    let tensor: Tensor3D = tensorBuffer.toTensor() as Tensor3D;

    await annotateImage(tensor, imagePath);
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