import { decodeImage, encodeImage } from "./tfjs-image-utils";
import type { Tensor3D } from "@tensorflow/tfjs-core";

const inputPath = "resources/face1.jpg";
decodeImage(inputPath)
    .then((tensor: Tensor3D) => {
        return encodeImage(tensor, inputPath, "decode-encode-demo")
    })
    .then(() => {
        console.log("Done");
    })
    .catch(err => {
        console.error(err);
    })