"""
Control ai2thor in interactive mode with a unity build path. Needs to be run from terminal.
If you don't provide a good path, it will try to download the 600+mb ai2thor assets.
"""

import argparse

import ai2thor.controller

parser = argparse.ArgumentParser(description='Run ai2thor in interactive mode')
parser.add_argument('--unity-build-path', type=str, default='ai2thor/unity/build-test.x86_64',
                    help='Path to a unity build with file ending in .x86_64')
args = parser.parse_args()

controller = ai2thor.controller.Controller()
controller.local_executable_path = args.unity_build_path
controller.start()

controller.step(dict(action='Initialize', gridSize=0.1, cameraY=-0.85, continuous=True))
event = controller.step(dict(action='Rotate', rotation=30))
controller.interact()
