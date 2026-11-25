const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const sharp = require('sharp');
const { predictImage } = require('./onnx-inference');

test('predictImage preprocesses RGB input and returns the frontend prediction fields', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'cancer-inference-'));
  const imagePath = path.join(directory, 'red.png');
  await sharp({
    create: {
      width: 2,
      height: 2,
      channels: 3,
      background: { r: 255, g: 0, b: 0 }
    }
  }).png().toFile(imagePath);

  let input;
  const session = {
    inputNames: ['image'],
    outputNames: ['logits'],
    async run(feeds) {
      input = feeds.image;
      return { logits: { data: Float32Array.from([-2]) } };
    }
  };

  try {
    const result = await predictImage(session, imagePath);
    assert.deepEqual(input.dims, [1, 3, 224, 224]);
    assert.ok(Math.abs(input.data[0] - ((1 - 0.485) / 0.229)) < 1e-5);
    assert.ok(Math.abs(input.data[224 * 224] - ((0 - 0.456) / 0.224)) < 1e-5);
    assert.equal(result.prediction, 'No Cancer');
    assert.ok(Math.abs(result.probability - (1 / (1 + Math.exp(2)))) < 1e-7);
    assert.equal(result.gradcamImage, null);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});

test('predictImage converts grayscale X-rays to three-channel input', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'cancer-inference-gray-'));
  const imagePath = path.join(directory, 'gray.png');
  await sharp({
    create: {
      width: 2,
      height: 2,
      channels: 3,
      background: { r: 128, g: 128, b: 128 }
    }
  }).greyscale().png().toFile(imagePath);

  let input;
  const session = {
    inputNames: ['image'],
    outputNames: ['logits'],
    async run(feeds) {
      input = feeds.image;
      return { logits: { data: Float32Array.from([2]) } };
    }
  };

  try {
    const result = await predictImage(session, imagePath);
    const normalized = (128 / 255 - 0.485) / 0.229;
    assert.deepEqual(input.dims, [1, 3, 224, 224]);
    assert.ok(Math.abs(input.data[0] - normalized) < 1e-5);
    assert.ok(Math.abs(input.data[224 * 224] - ((128 / 255 - 0.456) / 0.224)) < 1e-5);
    assert.equal(result.prediction, 'Cancer');
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});
