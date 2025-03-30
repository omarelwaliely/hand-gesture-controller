# Hand Gesture Controller

## Overview
Hand Gesture Controller is a Python-based project that allows players to control originally designed to play Minecraft using hand gestures. Utilizing the Mediapipe library for hand tracking and PyDirectInput for sending keyboard and mouse inputs, this project enables an interesting challenge.

## Features
- **Hand Gesture Recognition**: Detects various hand gestures using Mediapipe.
- **Keyboard & Mouse Integration**: Maps gestures to in-game controls for movement, actions, and interactions.
- **Customizable Mappings**: Modify gesture-to-key bindings easily.
- **Real-time Tracking**: Smooth and responsive tracking using threading for better control.

## Requirements
Ensure you have the following installed:
- Python 3.x
- OpenCV (`pip install opencv-python`)
- Mediapipe (`pip install mediapipe`)
- PyDirectInput (`pip install pydirectinput`)
- NumPy (`pip install numpy`)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/omarelwaliely/hand-gesture-controller.git
   cd minecraft-gesture-controller
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```

## Gesture Mappings
| Gesture | Action in Minecraft |
|---------|--------------------|
| Thumbs Up | Move Forward (W) |
| Thumbs Down | Move Backward (S) |
| OK Sign | Toggle Jump (Space) |
| Three Fingers Up | Toggle Sprint (R) |
| Index & Pinky Up | Scroll Up |
| Pinky Up | Scroll Down |
| Index, Middle, Pinky Up | Open Inventory (E) |
| Index, Ring, Pinky Up | Toggle Crouch (K) |
| Index Finger Up | Left Click |
| Index & Middle Up | Right Click |

## Customizing Gesture Mappings
To change gesture mappings, edit the `gesture_key_map` dictionary in `main.py`. Example:
```python
gesture_key_map = {
    'forward': ('w', 15),
    'backward': ('s', 15),
    'left click': ('left_click', 15),
    'scroll down': ('scroll_down', 10),
    'toggle jump': (['space'], -5),
    'right click': ('right_click', 5),
    'scroll up':('scroll_up', 10),
    'toggle sprint':(['r'],-5),
    'inventory':('e',10),
    'toggle crouch': ('k',-5)
}
```
Note that the numbers is how long between frames it allows you to perform the key again. For example if you do forward then change then do forward again within 15 frames it would not activate a second time, and negative sign is a toggle key rather than you having to keep holding up the sign. If you need to reconfigure anything it'll be this part since I had to use a 25 fps camera.

## Notes
- Ensure your camera is working correctly before launching the script.
- Adjust gesture sensitivity in the `detect_gesture` function if necessary.
- Close the application using the `Q` key.
- The crouch key and sprint key were set to 'k' and 'r' due to limitations on my machine, but you should be able to map it to 'lshift' and 'lctrl'

## License
This project is open-source and available under the MIT License.

## Author
Developed by Omar Elwaliely. Contributions and suggestions are welcome.

