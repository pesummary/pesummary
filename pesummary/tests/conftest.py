# Licensed under an MIT style license -- see LICENSE.md

"""Configuration for pytest.
"""

MARKERS = {
    "executabletest": "mark test as testing an executable script",
    "workflowtest": "mark test as testing a workflow",
    "ligoskymaptest": "mark test as testing ligo.skymap integration",
}


def pytest_configure(config):
    for mark, desc in MARKERS.items():
        config.addinivalue_line("markers", f"{mark}: {desc}")
