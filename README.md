Face Recognition Using Deep Learning
A real-time face recognition system that identifies and verifies people from images, videos, live
camera feeds and CCTV, built on a deep learning (ResNet) face-embedding pipeline.
The system supports all four recognition modes (1:1, 1:N, N:1 and N:N) and lets new people be
enrolled without retraining any model.
B.Tech Project, Department of Electronics and Communication Engineering, Netaji
Subhas University of Technology (NSUT), New Delhi Supervisor: Prof. Arti M.K.
Why this project
Passwords and keycards can be lost, shared or stolen, and manual identity checks are slow and
error-prone. Face recognition offers a faster, contactless and harder-to-fake way to verify
identity.
While surveying existing open-source face recognition projects, we found the same gaps again
and again:
Most support only one recognition mode (usually 1:1 or 1:N)
Many cannot process live video efficiently
Most work with a single input source (only images, or only a webcam)
Adding a new person often means retraining the whole model
Little or no testing in real-world conditions like low light or masks
This project was built to close those gaps in one system.
Key Features
Feature Description
Multi-mode
recognition
1:1 (verify one face against one reference), 1:N (identify one face from a
database), N:1 (verify many faces against one reference), N:N (identify
everyone in a frame)
Images, video files, live camera and CCTV streams
•
•
•
•
•
Feature Description
Multiple input
sources
Real-time
processing
Frame-by-frame recognition on live camera feeds
No-retrain
enrolment
New people are added by storing their face encodings, so no model training
is needed
Real-world
testing
Evaluated under normal conditions, low light and masked faces
How It Works
The system does not train a classifier for each person. Instead, it converts every face into a
128-dimensional embedding (a numeric "fingerprint" of the face) and compares embeddings
using distance.
1. Enrolment (adding a person)
flowchart TD
 A[Input image + name] --> B[Detect face]
 B --> C[Find 68 facial landmarks]
 C --> D[Generate 128-D encoding with ResNet]
 D --> E[(Store encoding + name in database)]
2. Recognition (image, video or camera)
flowchart TD
 A[Image / video frame / camera frame] --> B{Face detected?}
 B -- No --> X[No face detected]
 B -- Yes --> C[Generate 128-D encoding]
 C --> D[Compute distance to every stored encoding]
 D --> E{Distance < 0.4?}
 E -- Yes --> F[Known face: return name]
 E -- No --> G[Unknown face]
For live camera input, this loop runs on every frame until the user exits.
The models
The pipeline uses two pretrained models from the dlib library:
shape_predictor_68_face_landmarks : locates 68 key points on the face (eyes, nose,
mouth, jawline), which helps align the face before encoding
dlib_face_recognition_resnet_model_v1 : a ResNet network that converts an aligned
face into a 128-dimensional embedding. This pretrained model reports around 99.3%
accuracy on the LFW (Labeled Faces in the Wild) benchmark
Why a distance threshold of 0.4
Two embeddings of the same person sit close together, and embeddings of different people sit
far apart. The threshold decides how close is "close enough":
Lower threshold → stricter matching, fewer wrong matches, but more genuine users
rejected
Higher threshold → more lenient, fewer rejections, but higher risk of matching the wrong
person
A threshold of 0.4 was chosen as a strict setting suited to security use cases.
Tech Stack
Language: Python
Face detection, landmarks and embeddings: dlib (pretrained ResNet)
Image and video handling: OpenCV
Numerical computing: NumPy
Challenges and How We Approached Them
Challenge Impact Approach
Low light
Poor lighting weakens
face detection and feature
extraction
Illumination normalisation during pre-processing;
infrared cameras as a hardware option
Slow video
frame rate
Real-time recognition
needs fast processing per
frame
Model optimisation for faster inference; GPU
acceleration where available
•
•
•
•
•
•
•
•
Challenge Impact Approach
Masked
faces
Masks hide the nose and
mouth, which standard
models rely on
Analysed performance on masked faces;
masked-face datasets, 3D imaging and hybrid
biometrics (e.g. voice) identified as improvements
Comparison With Existing Systems
Feature Typical existing systems This project
One-to-One recognition Yes Yes
One-to-Many recognition Limited Yes
Many-to-One recognition Rare Yes
Many-to-Many recognition No Yes
Real-time camera processing Partial Yes
Video processing Rare Yes
CCTV integration No Yes
Add new users without retraining No Yes
Low-light handling Weak Tested
Masked-face handling Weak Analysed
Future Improvements
Fine-tune the embedding model on masked-face datasets to improve recognition with masks
Add liveness detection to block spoofing with printed photos or screens
Use a vector database for fast search when the number of enrolled people grows large
Wrap the system in a REST API and deploy it in a Docker container for easier integration
Team
Harshit Kumar (2022UEA6520)
Shrey Soni (2022UEA6505)
•
•
•
•
•
•
Supervisor: Prof. Arti M.K., Department of ECE, NSUT
Acknowledgements
dlib for the pretrained landmark and face recognition models
Labeled Faces in the Wild (LFW) benchmark dataset
OpenCV for image and video processing
•
•
• 
