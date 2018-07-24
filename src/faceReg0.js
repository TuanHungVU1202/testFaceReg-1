const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');

if (!cv.xmodules.face) {
  throw new Error('exiting: opencv4nodejs compiled without face module');
}

const basePath = '../imageData/face';
const imgsPath = path.resolve(basePath, 'imgs');
const nameMappings = ['hung', 'va', 'thu'];

const imgFiles = fs.readdirSync(imgsPath);

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT);
const getFaceImage = (grayImg) => {
  const faceRects = classifier.detectMultiScale(grayImg).objects;
  if (!faceRects.length) {
    throw new Error('failed to detect faces');
  }
  return grayImg.getRegion(faceRects[0]);
};

const images = imgFiles
  // get absolute file path
  .map(file => path.resolve(imgsPath, file))
  // read image
  .map(filePath => cv.imread(filePath))
  // face recognizer works with gray scale images
  .map(img => img.bgrToGray())
  // detect and extract face
  .map(getFaceImage)
  // face images must be equally sized
  .map(faceImg => faceImg.resize(80,80));

const isImageNine = (_, i) => imgFiles[i].includes('4');
const isNotImageNine = (_, i) => !isImageNine(_, i);
// use images 1 - 3 for training
const trainImages = images.filter(isNotImageNine);
// use images 4 for testing
const testImages = images.filter(isImageNine);
// make labels
const labels = imgFiles
  .filter(isNotImageNine)
  .map(file => nameMappings.findIndex(name => file.includes(name)));

const runPrediction = (recognizer) => {
  testImages.forEach((img) => {
    const result = recognizer.predict(img);
    console.log('predicted: %s, confidence: %s', nameMappings[result.label], result.confidence);
    cv.imshowWait('face', img);
    cv.destroyAllWindows();
  });
};

const eigen = new cv.EigenFaceRecognizer();
const fisher = new cv.FisherFaceRecognizer();
const lbph = new cv.LBPHFaceRecognizer();
eigen.train(trainImages, labels);
fisher.train(trainImages, labels);
lbph.train(trainImages, labels);

console.log('eigen:');
runPrediction(eigen);

console.log('fisher:');
runPrediction(fisher);

console.log('lbph:');
runPrediction(lbph);