const crypto = require('crypto');
const fs = require('fs');
const fsp = fs.promises;
const https = require('https');
const path = require('path');
const { Transform, pipeline } = require('stream');
const { promisify } = require('util');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

const pipelineAsync = promisify(pipeline);
const IMAGE_SIZE = 224;
const MAX_MODEL_BYTES = 250 * 1024 * 1024;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

function getResponse(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    if (target.protocol !== 'https:') {
      reject(new Error('Model URL must use HTTPS'));
      return;
    }

    const request = https.get(target, response => {
      const status = response.statusCode || 0;
      if (status >= 300 && status < 400 && response.headers.location) {
        response.resume();
        if (redirects >= 5) {
          reject(new Error('Model download exceeded the redirect limit'));
          return;
        }
        getResponse(new URL(response.headers.location, target).toString(), redirects + 1)
          .then(resolve, reject);
        return;
      }

      if (status !== 200) {
        response.resume();
        reject(new Error(`Model download failed with HTTP ${status}`));
        return;
      }

      resolve(response);
    });

    request.setTimeout(30000, () => {
      request.destroy(new Error('Model download timed out'));
    });
    request.on('error', reject);
  });
}

async function hashFile(filePath) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(filePath)) {
    hash.update(chunk);
  }
  return hash.digest('hex');
}

async function removeIfExists(filePath) {
  try {
    await fsp.unlink(filePath);
  } catch (error) {
    if (error.code !== 'ENOENT') {
      throw error;
    }
  }
}

async function downloadModel(url, destinationPath, expectedSha256) {
  if (!/^[a-f0-9]{64}$/i.test(expectedSha256 || '')) {
    throw new Error('ONNX_MODEL_SHA256 must be a 64-character SHA-256 digest');
  }

  const expected = expectedSha256.toLowerCase();
  await fsp.mkdir(path.dirname(destinationPath), { recursive: true });

  try {
    if ((await hashFile(destinationPath)) === expected) {
      return;
    }
    await fsp.unlink(destinationPath);
  } catch (error) {
    if (error.code !== 'ENOENT') {
      throw error;
    }
  }

  const temporaryPath = `${destinationPath}.${process.pid}.tmp`;
  await removeIfExists(temporaryPath);
  const hash = crypto.createHash('sha256');
  let downloadedBytes = 0;
  const verifyStream = new Transform({
    transform(chunk, encoding, callback) {
      downloadedBytes += chunk.length;
      if (downloadedBytes > MAX_MODEL_BYTES) {
        callback(new Error('Downloaded model exceeds the 250 MB limit'));
        return;
      }
      hash.update(chunk);
      callback(null, chunk);
    }
  });

  try {
    const response = await getResponse(url);
    await pipelineAsync(
      response,
      verifyStream,
      fs.createWriteStream(temporaryPath, { flags: 'wx' })
    );

    if (hash.digest('hex') !== expected) {
      throw new Error('Downloaded model SHA-256 does not match ONNX_MODEL_SHA256');
    }

    await fsp.rename(temporaryPath, destinationPath);
  } catch (error) {
    try {
      await removeIfExists(temporaryPath);
    } catch (cleanupError) {
      console.error('Failed to remove incomplete model download:', cleanupError);
    }
    throw error;
  }
}

async function createSession(modelPath) {
  return ort.InferenceSession.create(modelPath, {
    executionProviders: ['cpu'],
    intraOpNumThreads: 1,
    interOpNumThreads: 1
  });
}

async function predictImage(session, imagePath) {
  const { data, info } = await sharp(imagePath)
    .resize(IMAGE_SIZE, IMAGE_SIZE, { fit: 'fill', kernel: sharp.kernel.linear })
    .toColourspace('srgb')
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  if (info.channels !== 3) {
    throw new Error(`Expected a 3-channel image after preprocessing; received ${info.channels}`);
  }

  const planeSize = IMAGE_SIZE * IMAGE_SIZE;
  const input = new Float32Array(3 * planeSize);
  for (let pixel = 0; pixel < planeSize; pixel += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      input[channel * planeSize + pixel] =
        (data[pixel * 3 + channel] / 255 - MEAN[channel]) / STD[channel];
    }
  }

  const tensor = new ort.Tensor('float32', input, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const logit = Number(outputs[session.outputNames[0]].data[0]);
  if (!Number.isFinite(logit)) {
    throw new Error('ONNX model returned a non-finite prediction');
  }

  const probability = 1 / (1 + Math.exp(-logit));
  return {
    prediction: probability > 0.5 ? 'Cancer' : 'No Cancer',
    probability,
    confidence: Math.max(probability, 1 - probability),
    gradcamImage: null
  };
}

module.exports = { createSession, downloadModel, predictImage };
