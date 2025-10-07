AE667_Assignment2/
│
├── src/
│   ├── performance_estimator/
│   │   ├── __init__.py
│   │   └── models.py                 # All component models (rotor, fuselage, etc.)
│   │
│   ├── mission_planner/
│   │   ├── __init__.py
│   │   └── segments.py               # Mission segment calculations (cruise, loiter)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── atmosphere.py             # Standard atmosphere model
│   │   └── data_loader.py            # Loads airfoil data and vehicle JSON configs
│   │
│   └── analysis.py                   # Main script for running all assignment tasks
│
├── configs/
│   ├── team_helicopter.json          # Group's common helicopter design data
│   └── individual_helicopter.json    # Your personal helicopter design data
│
├── data/
│   └── airfoils/
│       └── naca_xxxx.csv             # Airfoil coefficient data
│
├── output/
│   ├── plots/                        # Directory for saved graphs
│   └── tables/                       # Directory for saved results (e.g., trim tables as CSV)
│
├── docs/
│   ├── algorithm_flowcharts.drawio   # Flow diagrams
│   └── user_manual.md                # User manual
│
├── .venv/                            # Python virtual environment folder
├── README.md                         # How to set up and run the code [cite: 136]
└── requirements.txt                  # Python libraries (numpy, scipy, matplotlib)