def detect_cheating(name, gaze):
    if name == "Unknown" or gaze != "Looking Center":
        return True
    return False