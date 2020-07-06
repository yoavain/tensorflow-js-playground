import { Tensor3D } from "@tensorflow/tfjs-node-gpu";
import { promises as fs } from "fs";
import * as tf from "@tensorflow/tfjs-node-gpu";
import { encodePng } from "@tensorflow/tfjs-node-gpu/dist/image";
import path from "path";

export const decodeImage = async (inputImagePath: string): Promise<Tensor3D> => {
    console.log(`Decoding image ${inputImagePath}`);
    const image: Buffer = await fs.readFile(inputImagePath);
    return tf.node.decodeImage(new Uint8Array(image), 3) as Tensor3D;
}

export const encodeImage = async (output: Tensor3D, inputImagePath: string, tag?: string): Promise<void> => {
    try {
        const nameTag: string = tag || "output";
        const outputPath = path.join(__dirname, "../output", `${path.parse(inputImagePath).name}-${nameTag}.png`);
        console.log(`Encoding image ${outputPath}`);
        const uint8Array: Uint8Array = await encodePng(output);
        await fs.writeFile(outputPath, uint8Array);
        console.log(`Image saved as ${outputPath}\r\n`);
    } catch (err) {
        console.error(err);
    }
}
