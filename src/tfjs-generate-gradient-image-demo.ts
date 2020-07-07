import * as tf from "@tensorflow/tfjs-node-gpu";
import { encodeImage } from "./tfjs-image-utils";

export const main = async (dim: number) => {
    let shape: [number, number, number] = [dim,dim,1];
    let val = Array(shape[0]*shape[1]*shape[2]).fill(0);
    for (let i = 0; i < val.length; i++) {
        val[i] = Math.floor(256 * i / val.length);
    }
    let uint8Array: Uint8Array = new Uint8Array(val);
    let tensor3d = tf.tensor3d(uint8Array, shape);
    encodeImage(tensor3d, "gradient.png", `${dim}x${dim}`)
        .then(() => console.log("Done"))
}

// run
if (process.argv.length < 3) {
    console.log('please pass an image to process. ex:');
    console.log('   ts-node tfjs-generate-gradient-image-demo.ts <number>');
}
else {
    let dimension: number = Number.parseInt(process.argv[2]);
    main(dimension).then(() => {})
}