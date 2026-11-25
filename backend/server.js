// Load environment variables
require('dotenv').config();

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;
const { existsSync } = require('fs');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 5000;
const PROJECT_ROOT = path.join(__dirname, '..');
const DEFAULT_CHECKPOINT_PATH = path.join(PROJECT_ROOT, 'Models', 'best_model.pth');
const DEFAULT_CONFIG_PATH = path.join(PROJECT_ROOT, 'ai', 'configs', 'config.yaml');

function getInferencePaths() {
  return {
    checkpointPath: process.env.CHECKPOINT_PATH
      ? path.resolve(__dirname, process.env.CHECKPOINT_PATH)
      : DEFAULT_CHECKPOINT_PATH,
    configPath: process.env.CONFIG_PATH
      ? path.resolve(__dirname, process.env.CONFIG_PATH)
      : DEFAULT_CONFIG_PATH
  };
}

function getPythonCommand() {
  const virtualEnvironments = process.platform === 'win32'
    ? [
        path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe'),
        path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe'),
        path.join(PROJECT_ROOT, 'ai', '.venv', 'Scripts', 'python.exe')
      ]
    : [
        path.join(PROJECT_ROOT, 'venv', 'bin', 'python'),
        path.join(PROJECT_ROOT, 'ai', '.venv', 'bin', 'python')
      ];

  return process.env.PYTHON_PATH
    || virtualEnvironments.find(existsSync)
    || (process.platform === 'win32' ? 'python' : 'python3');
}

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads');
    try {
      await fs.mkdir(uploadDir, { recursive: true });
      cb(null, uploadDir);
    } catch (error) {
      cb(error);
    }
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only .png, .jpg and .jpeg format allowed!'));
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Helper function to run Python inference
function runPythonInference(imagePath, checkpointPath, configPath) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, '..', 'ai', 'main.py');
    
    const pythonCommand = getPythonCommand();
    const pythonProcess = spawn(pythonCommand, [
      pythonScript,
      'inference',
      '--config', configPath,
      '--checkpoint', checkpointPath,
      '--image', imagePath,
      '--device', 'auto',
      '--visualize'  // Enable Grad-CAM generation
    ], {
      env: {
        ...process.env,
        MPLBACKEND: process.env.MPLBACKEND || 'Agg'
      }
    });

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });

    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python inference (${pythonCommand}): ${error.message}`));
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python process exited with code ${code}\n${errorData}`));
      } else {
        resolve(outputData);
      }
    });
  });
}

// Parse Python output to extract prediction results
function parsePythonOutput(output) {
  const lines = output.split('\n');
  const result = {
    prediction: 'Unknown',
    probability: 0,
    confidence: 0,
    gradcamPath: null
  };

  lines.forEach(line => {
    if (line.includes('Prediction:')) {
      const match = line.match(/Prediction:\s*(\w+\s*\w*)/);
      if (match) result.prediction = match[1].trim();
    }
    if (line.includes('Probability:')) {
      const match = line.match(/Probability:\s*([\d.]+)/);
      if (match) result.probability = parseFloat(match[1]);
    }
    if (line.includes('Confidence:')) {
      const match = line.match(/Confidence:\s*([\d.]+)/);
      if (match) result.confidence = parseFloat(match[1]);
    }
    if (line.includes('Visualization saved:')) {
      const match = line.match(/Visualization saved:\s*(.+)/);
      if (match) result.gradcamPath = match[1].trim();
    }
  });

  return result;
}

// Routes
app.get('/', (req, res) => {
  res.json({
    message: 'Cancer Detection API',
    version: '1.0.0',
    endpoints: {
      health: 'GET /health',
      predict: 'POST /predict',
      status: 'GET /status'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Main prediction endpoint
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    const imagePath = req.file.path;
    
    const { checkpointPath, configPath } = getInferencePaths();

    // Check if checkpoint exists
    try {
      await fs.access(checkpointPath);
    } catch {
      await fs.unlink(imagePath); // Clean up uploaded file
      return res.status(500).json({ 
        error: 'Model checkpoint not found',
        message: 'Check CHECKPOINT_PATH or place a model checkpoint at Models/best_model.pth',
        expectedPath: checkpointPath,
      });
    }

    // Run inference
    console.log(`Processing image: ${imagePath}`);
    const output = await runPythonInference(imagePath, checkpointPath, configPath);
    const result = parsePythonOutput(output);

    // Read Grad-CAM image if generated
    let gradcamBase64 = null;
    if (result.gradcamPath) {
      try {
        const gradcamData = await fs.readFile(result.gradcamPath);
        gradcamBase64 = `data:image/png;base64,${gradcamData.toString('base64')}`;
        // Clean up Grad-CAM file
        await fs.unlink(result.gradcamPath);
      } catch (error) {
        console.error('Error reading Grad-CAM:', error);
      }
    }

    // Clean up uploaded file
    await fs.unlink(imagePath);

    res.json({
      success: true,
      result: {
        prediction: result.prediction,
        probability: result.probability,
        confidence: result.confidence,
        gradcamImage: gradcamBase64,
        message: result.prediction === 'Cancer' 
          ? 'Cancer detected - Please consult a medical professional immediately' 
          : 'No cancer detected - Routine follow-up recommended'
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Prediction error:', error);
    
    // Clean up uploaded file if it exists
    if (req.file) {
      try {
        await fs.unlink(req.file.path);
      } catch (unlinkError) {
        console.error('Failed to delete uploaded file:', unlinkError);
      }
    }

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Batch prediction endpoint
app.post('/predict-batch', upload.array('images', 10), async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No image files uploaded' });
    }

    const { checkpointPath, configPath } = getInferencePaths();

    const results = [];

    for (const file of req.files) {
      try {
        const output = await runPythonInference(file.path, checkpointPath, configPath);
        const result = parsePythonOutput(output);
        
        results.push({
          filename: file.originalname,
          prediction: result.prediction,
          probability: result.probability,
          confidence: result.confidence
        });

        await fs.unlink(file.path);
      } catch (error) {
        results.push({
          filename: file.originalname,
          error: error.message
        });
      }
    }

    res.json({
      success: true,
      results: results,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Batch prediction error:', error);
    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Status endpoint
app.get('/status', async (req, res) => {
  const { checkpointPath } = getInferencePaths();
  
  let modelAvailable = false;
  try {
    await fs.access(checkpointPath);
    modelAvailable = true;
  } catch {
    modelAvailable = false;
  }

  res.json({
    status: 'running',
    modelAvailable: modelAvailable,
    checkpointPath: checkpointPath,
    timestamp: new Date().toISOString()
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    error: err.message,
    timestamp: new Date().toISOString()
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`\n🚀 Cancer Detection API Server`);
  console.log(`📍 Server running on http://localhost:${PORT}`);
  console.log(`\n📋 Available endpoints:`);
  console.log(`   GET  /          - API information`);
  console.log(`   GET  /health    - Health check`);
  console.log(`   POST /predict   - Single image prediction`);
  console.log(`   POST /predict-batch - Batch prediction`);
  console.log(`   GET  /status    - Server status\n`);
});