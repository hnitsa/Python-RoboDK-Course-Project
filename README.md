This project implements a machine vision pipeline which converts an object image into executable robot motion. The system detects the contour of an object from a photo, transforms the detected outline into real world coordinates, and generates a robot program which reproduces the shape. The workflow combines image processing, geometric scaling, and robotic motion generation. The resulting path loads into RoboDK and produces robot code for the YASKAWA Motoman HC10DTP. The robot then follows the detected contour and draws the object outline.

Project Functionality

• Image preprocessing. The program loads an input image and converts it to grayscale. Gaussian blur and thresholding remove noise and separate the object from the background.

• Paper detection. The algorithm locates the sheet of paper in the image and extracts the region of interest. This step removes background objects and improves detection accuracy.

• Scale detection. A blue reference line with a known real length appears in the image. The system detects this line and calculates the pixel to millimeter ratio.

• Object contour extraction. The program detects the main object using contour detection. The contour is simplified and converted into a set of ordered points.

• Coordinate conversion. The detected pixel coordinates convert into millimeter coordinates using the scale factor. The points shift relative to the object center so the robot follows the correct path.

• Robot path generation. The contour points convert into robot movements. Travel movements use higher speed and drawing movements use lower speed.

• Robot program export. The system generates a robot program in the format supported by the robot controller and loads it inside RoboDK for simulation or execution.

Use Instructions

Install required libraries.

Python environment requires
• OpenCV
• NumPy
• RoboDK Python API

Prepare the input image.

• Place the object on a sheet of paper.
• Draw a blue reference line with known length at the bottom of the image.
• Take a photo with the camera.

Run the Python script.

The script performs the following operations.

• Detects the paper region.
• Finds the blue reference line and calculates scale.
• Extracts the object contour.
• Converts contour points into millimeter coordinates.

Open RoboDK simulation.

The script connects to RoboDK and creates a robot program automatically. The generated program contains motion commands which follow the detected contour.

Generate robot code.

Inside RoboDK export the robot program to the controller language. The output file contains linear motion instructions which trace the object outline.

Transfer the program to the robot.

Upload the generated program file to the robot controller through USB or network connection. Start the program on the robot controller. The robot follows the generated trajectory and reproduces the detected outline.
