# suspension-kinematics

## Installation

### Basic installation
For core kinematics functionality without visualization dependencies:

```
pip install kinematics
```

### Full installation with visualization
For complete functionality including animation generation:

```
pip install kinematics[viz]
```

## Usage examples

Use of [uv](https://github.com/astral-sh/uv) is generally advised when running the tool, e.g.,

````
uv run kinematics <command> [options]
````   

### Visualization of a geometry at design condition:

```
kinematics visualize --geometry=tests/data/geometry.yaml --output=plot.png
```

### Basic sweep with CSV export:

```
kinematics --geometry geometry.yaml --sweep sweep.yaml --out results.csv
```

### Full sweep with Parquet export and animation:

```
kinematics --geometry geometry.yaml --sweep sweep.yaml --out results.parquet --animation-out animation.mp4
```

Note: If you try to use visualization features without installing the [viz] extra, you will receive an error indicating that the required visualization dependencies are not installed.


