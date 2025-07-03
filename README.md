# Polychase

A free and open-source motion tracking addon for Blender, inspired by KeenTools GeoTracker.

## Overview

Polychase is a 3D motion tracking solution that allows you to track camera movement or object motion in video footage within Blender. It uses optical flow analysis and PnP, aided by user input to provide accurate tracking results.

## Features

### Core Tracking Capabilities
- **3D Pin Mode**: Place and manage tracking pins on 3D geometry
- **Camera/Geometry Tracking**: Track camera/geometry movement through 3D space
- **Trajectory Refinement**: Refine tracking results using bundle adjustment

### Advanced Features
- **Variable Camera Parameters**: Support for estimating focal length and principal point
- **Keyframe Management**: Complete keyframe control for tracked animation
- **Scene Transformation**: Transform entire tracked scenes
- **Animation Conversion**: Convert between camera and object tracking
- **Real-time Preview**: Live tracking progress and results
- **Mask Support**: 3D masking for selective tracking

### User Interface
- **Integrated Blender UI**: Native Blender panels and operators
- **Visual Feedback**: Color-coded pins, wireframes, and progress indicators
- **Customizable Appearance**: Adjustable pin colors, sizes, and wireframe styles

## Requirements

- **Blender**: 4.2.0 or higher
- **Operating System**: Windows / Linux
- **Build Dependencies**: 
  - OpenCV (for computer vision operations)
  - Eigen (for linear algebra)
  - Embree (for ray casting acceleration)
  - pybind11 (for Python bindings)

## Installation

### Prerequisites

Make sure you have the following installed:
- CMake 3.15 or higher
- C++20 compatible compiler
- Python 3.x with development headers

### Building from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/theartful/polychase.git
   cd polychase
   git submodule update --init --recursive
   ```

2. **Set up Python virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install pybind11-stubgen
   ```

3. **Build the project:**
   ```bash
   mkdir build
   cd build
   cmake -DCMAKE_INSTALL_PREFIX=<install-prefix> ..
   make -j$(nproc)
   cd ..
   ```

4. **Generate Python stubs after building:**
   ```bash
   ../generate_stubs.sh
   ```

5. **Install the Blender addon:**
   - Copy the installed `blender_addon` folder to your Blender addons directory

## Usage

### Basic Workflow

1. **Setup Scene:**
   - Import your video footage as a movie clip
   - Add or import the 3D geometry you want to track
   - Set up a camera object

2. **Create Tracker:**
   - Open the Polychase panel in Blender's 3D viewport
   - Create a new tracker
   - Assign your clip, geometry, and camera

3. **Analyze Video:**
   - Set the database path for optical flow storage
   - Run "Analyze Video" to generate optical flow data

4. **Pin Mode:**
   - Enter pin mode to place tracking points on your 3D geometry
   - Add pins by clicking on the geometry surface
   - Drag the pins to adjust the pose of the geometry/camera

5. **Track Sequence:**
   - Choose tracking direction (forward/backward)
   - Select tracking target (camera or geometry)
   - Run tracking to generate keyframes

6. **Refine Results:**
   - Use the refine sequence tool to improve tracking accuracy

### Pin Mode Controls

- **Left Click**: Add new pin
- **Right Click**: Delete pin
- **M**: Go to mask drawing mode
- **ESC**: Exit pin mode

## Technical Details

### Architecture

- **C++ Core**: High-performance tracking algorithms written in C++
- **Python Bindings**: pybind11 integration for Blender compatibility  
- **Blender Integration**: Native Blender addon with custom operators and panels

### Algorithms

- **Optical Flow**: Off-the-shelf OpenCV solution
- **3D Tracking**: PnP (Perspective-n-Point) solving for camera pose estimation
- **Bundle Adjustment**: Global non-linear optimization for trajectory refinement
- **Ray Casting**: Accelerated mesh intersection using Embree

### Demo & Technical Walkthrough
[![Watch the technical walkthrough on YouTube](https://img.youtube.com/vi/W4HNmcjFuLw/hqdefault.jpg)](https://youtu.be/W4HNmcjFuLw)

## Contributing

We welcome contributions! Feel free to:
- Report bugs and request features via GitHub Issues
- Submit pull requests with improvements
- Help with documentation and testing

By contributing to this project, you agree that your contributions
may be relicensed under more permissive licenses (such as MIT or Apache-2.0) in
the future, in addition to the current GPL-3.0-or-later license.

**Note**: This is alpha software under active development. Please backup your projects before use and report any issues you encounter. 
