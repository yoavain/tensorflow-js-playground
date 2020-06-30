import * as tf from '@tensorflow/tfjs-node-gpu';
import { GraphModel, Rank, Tensor } from '@tensorflow/tfjs-node-gpu';
import { promises as fs } from 'fs';
import * as  maxvis from '@codait/max-vis';
import * as path from "path";
import { labels } from "./tfjs-image-annotation-demo-labels";

const modelUrl = 'https://tfhub.dev/tensorflow/tfjs-model/ssdlite_mobilenet_v2/1/default/1';
const maxNumBoxes = 8;

// load COCO-SSD graph model
const loadModel = async (): Promise<GraphModel> => {
    return await tf.loadGraphModel(modelUrl, {fromTFHub: true});
}

// convert image to Tensor
const processInput = async (imagePath: string): Promise<Tensor<Rank>> => {
    console.log(`preprocessing image ${imagePath}`);
    const image: Buffer = await fs.readFile(imagePath);
    return tf.node.decodeImage(new Uint8Array(image), 3).expandDims();
}

// run prediction with the provided input Tensor
const runModel = (model: GraphModel, inputTensor: Tensor<Rank>): Promise<Tensor | Tensor[]> => {
    console.log('running model');
    return model.executeAsync(inputTensor);
}

// process the model output into a friendly JSON format
const processOutput = (prediction, width, height) => {
    console.log('processOutput');

    const [maxScores, classes] = extractClassesAndMaxScores(prediction[0]);
    const indexes = calculateNMS(prediction[1], maxScores);

    return createJSONresponse(prediction[1].dataSync(), maxScores, indexes, classes, width, height);
}

// determine the classes and max scores from the prediction
const extractClassesAndMaxScores = predictionScores => {
    console.log('calculating classes & max scores');

    const scores = predictionScores.dataSync();
    const numBoxesFound = predictionScores.shape[1];
    const numClassesFound = predictionScores.shape[2];

    const maxScores = [];
    const classes = [];

    // for each bounding box returned
    for (let i = 0; i < numBoxesFound; i++) {
        let maxScore = -1;
        let classIndex = -1;

        // find the class with the highest score
        for (let j = 0; j < numClassesFound; j++) {
            if (scores[i * numClassesFound + j] > maxScore) {
                maxScore = scores[i * numClassesFound + j];
                classIndex = j;
            }
        }

        maxScores[i] = maxScore;
        classes[i] = classIndex;
    }

    return [maxScores, classes];
}

// perform non maximum suppression of bounding boxes
const calculateNMS = (outputBoxes, maxScores) => {
    console.log('calculating box indexes');

    const boxes = tf.tensor2d(outputBoxes.dataSync(), [outputBoxes.shape[1], outputBoxes.shape[3]]);
    const indexTensor = tf.image.nonMaxSuppression(boxes, maxScores, maxNumBoxes, 0.5, 0.5);

    return indexTensor.dataSync();
}

// create JSON object with bounding boxes and label
const createJSONresponse = (boxes, scores, indexes, classes, width: number, height: number) => {
    console.log('create JSON output');

    const count = indexes.length;
    const objects = [];

    for (let i = 0; i < count; i++) {
        const bbox = [];

        for (let j = 0; j < 4; j++) {
            bbox[j] = boxes[indexes[i] * 4 + j];
        }

        const minY = bbox[0] * height;
        const minX = bbox[1] * width;
        const maxY = bbox[2] * height;
        const maxX = bbox[3] * width;

        objects.push({
            bbox: [minX, minY, maxX, maxY],
            label: labels[classes[indexes[i]]],
            score: scores[indexes[i]]
        });
    }

    return objects;
}

const annotateImage = async (prediction, imagePath) => {
    console.log(`annotating prediction result(s)`);
    const annotatedImageBuffer = await maxvis.annotate(prediction, imagePath);
    const f = path.join(__dirname, "../output", `${path.parse(imagePath).name}-annotate.png`);
    try {
        await fs.writeFile(f, annotatedImageBuffer);
        console.log(`annotated image saved as ${f}\r\n`);
    }
    catch (err) {
        console.error(err);
    }
}

// run
if (process.argv.length < 3) {
    console.log('please pass an image to process. ex:');
    console.log('   ts-node tfjs-model.ts /path/to/image.jpg');
}
else {
    let imagePath = process.argv[2];
    let width: number = 1;
    let height: number = 1;
    loadModel()
        .then(async (model: GraphModel) => {
            const inputTensor: Tensor<Rank> = await processInput(imagePath);
            height = inputTensor.shape[1];
            width = inputTensor.shape[2];
            return runModel(model, inputTensor);
        })
        .then((prediction: Tensor<Rank> | Tensor[]) => {
            const jsonOutput = processOutput(prediction, width, height);
            console.log(jsonOutput);
            annotateImage(jsonOutput, imagePath);
        })
        .catch(err => console.log(err));
}