import depthai as dai

# Connect to the device and query its hardware
try:
    with dai.Device() as device:
        print(f"Device Name: {device.getDeviceName()}")
        print("-" * 30)

        # Get detailed features for all connected sensors
        features = device.getConnectedCameraFeatures()

        if not features:
            print("No cameras detected.")
        else:
            print("Detected Sensors:")
            for cam in features:
                print(f" - Socket: {cam.socket}")
                print(f" - Sensor Name: {cam.sensorName}")
                print(f" - Supported Types: {cam.supportedTypes}")
                print("-" * 30)

except RuntimeError as e:
    print(f"Could not connect to the camera. Ensure it is plugged in. Error: {e}")
