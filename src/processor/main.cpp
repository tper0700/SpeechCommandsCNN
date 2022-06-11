#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>

#include <opencv2/dnn/dnn.hpp>

#include <iostream>

#define DEFAULT_FILE "wow_test.wav"
#define MODEL_FILE "../../../data/M5.onnx"

#define AUDIO_RATE 4000

float pBuffer[AUDIO_RATE] = {0};

const char * const LABELS[] = {
"backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"
};

using namespace std;

// loadSound: Load sound file using DrWav and return as a downsampled array.
// Reuses the same static buffer, can only process one sound a a time.
float* loadSound(const char * const sound = DEFAULT_FILE) {
  unsigned int channels;
  unsigned int sampleRate;
  drwav_uint64 totalPCMFrameCount;

  cout << "Loading Sound " << sound << endl;

  float* pSampleData = drwav_open_file_and_read_pcm_frames_f32(
    sound,
    &channels,
    &sampleRate,
    &totalPCMFrameCount,
    NULL
  );


  if (pSampleData == NULL) {
    printf("Error loading file\n");
    return NULL;
  }

  cout << "Sample Rate: " << sampleRate << endl;
  cout << "   Channels: " << channels << endl;
  cout << " PCM Frames: " << totalPCMFrameCount << endl;

  for (int i = 0; i < totalPCMFrameCount && i < 16000; i += 4) {
    pBuffer[i / 4] = (
      pSampleData[i] +
      pSampleData[i + 1] +
      pSampleData[i + 2] +
      pSampleData[i + 3]) / 4.0;
  }

  for (int i = 0; i < 10; i++) {
    cout << i << ": " << pSampleData[i] << " : " << pBuffer[i] << endl;
  }

  drwav_free(pSampleData, NULL);
  return pBuffer;
}

// getPrediction: Use OpenCV model to infer the spoken word.
int getPrediction(float* pSound) {
  cout << "Loading Net file" << endl;
  cv::dnn::Net net = cv::dnn::readNet(MODEL_FILE);

  cout << "Making Matrix" << endl;
  int sz[] = {1, 1, AUDIO_RATE};
  cv::Mat M(3, sz, CV_32FC1, pSound);
  int szP[] = {AUDIO_RATE};
  cv::Mat Slice = M.reshape(1,1, szP);
  cout << "Mat:\n" << Slice.rowRange(0,10) << endl;

  cout << "Setting input into net" << endl;
  net.setInput(M);

  cout << "Making Prediction" << endl;
  cv::Mat prob = net.forward();
  int szO[] = {35};
  cv::Mat output = prob.reshape(1, 1, szO);

  cout << "Prediction:\n" << output << endl;
  int maxIdx = -1;

  cv::minMaxLoc(cv::SparseMat(output), NULL, NULL, NULL, &maxIdx);
  cout << "Prediction index: " << maxIdx << endl;

  return maxIdx;
}

int main() {
  cout << "Running Test" << endl;
  float* pSound = loadSound();
  if (pSound == NULL) { return -1; }

  int result = getPrediction(pSound);
  cout << "Predicted class: " << LABELS[result] << endl;

  return 0;
}
