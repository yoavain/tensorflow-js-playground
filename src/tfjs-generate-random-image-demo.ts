import * as tf from "@tensorflow/tfjs-node-gpu";
import { Rank, Tensor3D, TensorBuffer } from "@tensorflow/tfjs-node-gpu";
import { encodeImage } from "./tfjs-image-utils";

const main = async (dimension: number) => {
    let inputShape: [number, number, number] = [dimension, dimension, 3];
    let tensorBuffer: TensorBuffer<Rank, "int32"> = tf.buffer(inputShape, "int32");

    for (let x = 0; x < dimension; x++) {
        for (let y = 0; y < dimension; y++) {
            for (let c = 0; c < 3; c++) {
                tensorBuffer.set(x, y, c, Math.floor(Math.random() * 256 * 256 * 256));
            }
        }
    }

    let r3Tensor: Tensor3D = tf.tensor3d(new Uint8Array(tensorBuffer.values), inputShape);
    await encodeImage(r3Tensor, "./resources/image.png", "random");
};

// run
if (process.argv.length < 3) {
    console.log('please pass an image to process. ex:');
    console.log('   ts-node tfjs-generate-random-image-demo.ts <number>');
}
else {
    let dimension: number = Number.parseInt(process.argv[2]);
    main(dimension).then(() => {})
}