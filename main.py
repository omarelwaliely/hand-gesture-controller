from controller import HandGestureControl

controller = HandGestureControl(max_num_hands=2, movement_multiplier=2.5, required_consecutive_frames=10)

controller.start()