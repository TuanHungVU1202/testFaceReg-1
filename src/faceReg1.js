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

const trainImgs = imgFiles
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

  // make labels
const labels = imgFiles
  .map(file => nameMappings.findIndex(name => file.includes(name)));

const lbph = new cv.LBPHFaceRecognizer();
lbph.train(trainImgs, labels);

const twoFacesImg = cv.imread(path.resolve(basePath, 'test1.jpg'));
const result = classifier.detectMultiScale(twoFacesImg.bgrToGray());

const minDetections = 10;
result.objects.forEach((faceRect, i) => {
  if (result.numDetections[i] < minDetections) {
    return;
  }
  const faceImg = twoFacesImg.getRegion(faceRect).bgrToGray();
  const who = nameMappings[lbph.predict(faceImg).label];

  const rect = cv.drawDetection(
    twoFacesImg,
    faceRect,
    { color: new cv.Vec(255, 0, 0), segmentFraction: 4 }
  );

  const alpha = 0.4;
  cv.drawTextBox(
    twoFacesImg,
    new cv.Point(rect.x, rect.y + rect.height + 10),
    [{ text: who }],
    alpha
  );
});

cv.imshowWait('result', twoFacesImg);