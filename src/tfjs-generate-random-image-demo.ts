import { randomNormal, Tensor3D } from "@tensorflow/tfjs-node-gpu";
import { encodeImage } from "./tfjs-image-utils";

const main = async (dimension: number) => {
    const inputShape: [number, number, number] = [dimension, dimension, 3];
    let randomTensor: Tensor3D = randomNormal(inputShape, 128, 50, "int32");
    await encodeImage(randomTensor, "./resources/image.png", "random");
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