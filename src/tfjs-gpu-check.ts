import * as tf from '@tensorflow/tfjs-node-gpu';
import type { Sequential } from '@tensorflow/tfjs-node-gpu';

// Train a simple model:
const model: Sequential = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
    epochs: 100,
    callbacks: {
        onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
    }
});
