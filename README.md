


*****  Vehicle Detection and Tracking with Enhanced Features
Overview  *****

This project builds upon a previously developed vehicle detection system by introducing additional features such as vehicle ID, speed, and color. Using these features, the system not only detects vehicles but also tracks their speeds based on collected data.

🎯 Key Features


Vehicle Detection: Identifies vehicles in a given video feed.

Vehicle Tracking: Assigns unique IDs to each detected vehicle for effective tracking.

Speed Estimation: Calculates the speed of each vehicle using frame-by-frame movement analysis.

Color Classification: Detects and classifies the color of each vehicle.


📁 Files


car.mp4: Input video demonstrating the initial state (raw vehicle detection).

car_tracked_output.mp4: Output video showing the tracked vehicles with their IDs, speeds, and colors displayed.


🛠 Technologies Used


Python: Primary programming language.

OpenCV: For video processing and image analysis.

NumPy: For data handling and mathematical operations.

Machine Learning: Color classification model for vehicle detection.



🚀 How It Works

Preprocessing: The input video is processed frame by frame.

Vehicle Detection: Vehicles are detected using object detection algorithms (e.g., YOLO, Haar cascades).

Tracking and ID Assignment: Each vehicle is assigned a unique ID and tracked over consecutive frames.

Speed Estimation: Vehicle speed is calculated based on pixel displacement and frame rate.

Color Detection: The dominant color of the vehicle is identified and labeled.

🖥 Setup and Installation


🎥 Demo
Check out the transformation from simple detection to enhanced tracking with enriched features:

Initial State	Final State (Tracked Output)
	
📊 Use Cases


Traffic Monitoring: Analyze traffic flow and vehicle speeds in real time.

Law Enforcement: Identify overspeeding vehicles and other traffic violations.

Smart Cities: Integration with intelligent transportation systems.

🌟 Future Improvements


Integration with real-time traffic cameras.

Addition of lane detection and traffic rule violation features.

Enhanced speed estimation using GPS data.

👨‍💻 Author


Feel free to contact me for questions or collaborations!

GitHub: https://github.com/yusufafsar23


Email: yusufafsar7115@gmail.com


📜 License


This project is licensed under the MIT License.

