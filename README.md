# Tiny-Mocap

Tiny-Mocap is a motion capture and streaming solution. It facilitates the capture and streaming of motion data from IoT devices like Raspberry Pi and laptops. Additionally, it offers server-side functionality for processing and visualizing the captured motion and background.

## Components

The Tiny-Mocap repository consists of the following components:

### `udp_client.py`

The `udp_client.py` file serves as the capturing interface on IoT devices. It allows you to capture motion data from devices like Respabary Pi and send it over the network.

### `udp_server.py`

The `udp_server.py` file contains the server-side functions responsible for receiving the captured landmarks and applying inverse kinematics transformation to the received data. This component plays a crucial role in processing the motion data.

### `app.py`

The `app.py` file is a Flask server implementation that provides a web-based visualization of the streamed motion data. By running this server, you can visualize the received stream and gain insights from the captured motion data in a user-friendly manner.

## Usage

To use Tiny-Mocap, follow these steps:

1. Set up the IoT devices, ensuring they have the necessary hardware and software dependencies.
2. Run the `udp_server.py` file on the server-side to listen for incoming motion data.
3. Execute the `udp_client.py` file on the IoT devices to capture motion data and transmit it to the server.
4. Launch the `app.py` Flask server to visualize the received stream and explore the captured motion data through a web interface.

Make sure to customize and configure the components according to your specific requirements and network setup.

## Dependencies

Tiny-Mocap has the following dependencies:

- Python 3.x
- Flask
- Opencv-python
- Mediapipe

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions to Tiny-Mocap are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

Before contributing, please familiarize yourself with the [contribution guidelines](https://www.notion.so/yqg/CONTRIBUTING.md).

## License

## Acknowledgements

## Contact

For any questions or inquiries, please contact [insert contact information here].

Feel free to reach out if you need any assistance or have any feedback.

Enjoy using Tiny-Mocap!