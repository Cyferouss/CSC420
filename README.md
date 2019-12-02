# Squid Vision

In this episode of Spongebob, https://www.youtube.com/watch?v=kwXI3Lh1wwI Squidward shows Spongebob how noses are properly sculpted. Our mission is to create a snapchat like filter which converts incorrectly shaped noses within the real world to Squidward's standards.

## Prerequisites
We use OpenCV and Tensorflow with a keras back

## Running the program

### Using Parameters
To run this using parameters, 
python VideoSection.py [INPUT_VIDEO PATH] [OUTPUT_VIDEO PATH] [WEBCAM=MODE True/False]

Alternatively if you're using VS Code you can add something like the following to the launch.json
"args":["C:/Users/skunk/Desktop/csc420proj/CSC420/TestVideo.mp4", 
    "C:/Users/skunk/Desktop/csc420proj/CSC420/output.avi",
    "False"]

NOTE: ALL 3 OF THESE PARAMETERS ARE REQUIRED

### Running Without Parameters
Alternatively if you want you can manually edit the Configuration Files.
Load resources section of the code.

