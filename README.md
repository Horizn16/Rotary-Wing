# AE 667: Helicopter Performance and Mission Planning Tool

## Features

- **Calculate aerodynamic forces and moments** on the helicopter for given pilot inputs.
- **Solve for trim conditions** required for steady, level flight.
- **Generate power-vs-speed curves** and determine key mission performance metrics:
    - Maximum speed
    - Maximum endurance speed
    - Maximum range speed

## File Structure

The project is organized into a modular structure to separate data, source code, and results:

```
AE667_Assignment2/
├── Assignment 2.pdf                  # Assignment instructions
├── configs/                          # Helicopter design parameters (JSON)
│   ├── individual_helicopter.json
│   └── team_helicopter.json
├── data/                             # Airfoil data, etc.
├── docs/                             # Documentation
│   └── tree.md
├── Group_Assignment_2_Templete.pptx  # Group assignment template
├── Individual Assignment 2 Templete.pptx # Individual assignment template
├── output/                           # Generated plots and results
│   └── plots/
│       ├── pilot_input_tests.png
│       └── power_curve.png
├── README.md                         # This file
├── requirements.txt                  # Project dependencies
└── src/                              # All Python source code
    ├── analysis.py
    ├── mission_planner/
    │   ├── __init__.py
    │   └── segments.py
    ├── performance_estimator/
    │   ├── __init__.py
    │   └── models.py
    └── utils/
        ├── atmosphere.py
        ├── data_loader.py
        └── __init__.py
```

## Setup and Installation

Instructions are provided for both **Windows** and **Ubuntu (Linux)**.

### Prerequisites

- Python 3.8 or newer ([Download Python](https://www.python.org/downloads/))

---

### Windows

1. **Open Command Prompt or PowerShell.**
2. **Navigate to the project root directory:**
     ```
     cd AE667_Assignment2
     ```
3. **Create a Python virtual environment:**
     ```
     python -m venv venv
     ```
4. **Activate the virtual environment:**
     - PowerShell:
         ```
         .\venv\Scripts\Activate.ps1
         ```
     - Command Prompt:
         ```
         .\venv\Scripts\activate.bat
         ```
5. **Install required libraries:**
     ```
     pip install -r requirements.txt
     ```

---

### Ubuntu (Linux)

1. **Open a Terminal.**
2. **Navigate to the project root directory:**
     ```
     cd AE667_Assignment2
     ```
3. **(Optional) Install `python3-venv` if needed:**
     ```
     sudo apt-get install python3-venv
     ```
4. **Create a Python virtual environment:**
     ```
     python3 -m venv venv
     ```
5. **Activate the virtual environment:**
     ```
     source venv/bin/activate
     ```
6. **Install required libraries:**
     ```
     pip install -r requirements.txt
     ```

---

## How to Run the Code

1. **Activate your virtual environment.**
2. **Navigate to the `src` directory:**
     ```
     cd src
     ```
3. **Run analyses using the following commands:**

     - **Pilot Input Tests (Task 3):**
         ```
         python3 analysis.py --task pilot_test
         ```
     - **Trim Analysis (Task 4 & 7):**
         ```
         python3 analysis.py --task trim
         ```
     - **Mission Performance Analysis (Task 5 & 8):**
         ```
         python3 analysis.py --task mission
         ```

## Specifying a Helicopter Design

By default, the team helicopter design is used. To specify your individual design, add the `--config` flag to any command:

```
# Example: Run the mission analysis for the individual design
python3 analysis.py --task mission --config ../configs/individual_helicopter.json
```

## Output

- All generated plots and tables are saved automatically inside the `output/plots/` directory.
- Key results are printed directly to the terminal.