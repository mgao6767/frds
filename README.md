# FRDS


General purpose financial research data services.

## Usage

```python
python -m frds.run
```

## Test

```python
python -m unittest
```

## Structure

| Module          | Description                                        |
|-----------------|----------------------------------------------------|
| `frds.data`     | provides data access to various data vendors.      |
| `frds.ra`       | the hard-working RA that does all the computation. |
| `frds.measures` | the collection of measures to estimate.            |

## Measures

| Measure                                      | Description                    | Datasets Used |
|----------------------------------------------|--------------------------------|---------------|
| [tangibility](/frds/measures/tangibility.py) | Asset tangibility = ppent / at | wrds.funda    |