"""a_data.csv (raw data)
    ↓
create_a_data_source() → (raw_sample, timestamp)
    ↓
StreamingSimulator._processing_loop()
    ├─ sample = raw_sample                    ← RAW SIGNAL
    └─ feature_deriver.add_sample(raw_sample)
         ↓
    SmoothedSignalDeriver (calculates moving average)
         ↓
    feature = deriver.get_feature()           ← MODEL VALUE
    ↓
    change_point_detector.update(feature, raw_sample=sample)
         ↓
    ResidualThresholdDetector (calculates |raw - model|)
         ↓
    is_change = (residual > threshold)        ← ANOMALY FLAG
    ↓
Queue.put({"sample": sample, "feature": feature, "change_points": [...]})
    ↓
StreamingSimulator._update_plot()
    ├─ samples[]  = all raw samples
    ├─ features[] = all model values
    └─ change_points[] = anomaly timestamps
    ↓
ModelLineRenderer.update()
    ├─ line.set_data(times, samples)          ← Draws RAW line (black)
    ├─ model_line.set_data(times, features)   ← Draws MODEL line (red dashed)
    └─ vlines.set_data(...)                   ← Draws VERTICAL lines (purple)

"""

# -------------------------------------------------------------------
# set up logging before importing other modules
# -------------------------------------------------------------------

import logging
from streamsim.src.core.config import LoggingSetup, PlottingSetup
from streamsim.src.core.config import LOG_DIR

logging_setup = LoggingSetup(
    level=logging.INFO,
    log_file=f"{LOG_DIR}/model_demo.log",
    timestamp_filename=True,             # Set to True if you want unique filenames per run
    console=True
)
logging_setup.setup_logging()
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# import main components for this demo
# -------------------------------------------------------------------

# Data Source: Reads from a CSV file with 'timestamp' and 'value' columns
from streamsim.src.sources.a_data_source import create_a_data_source
# deriver = SmoothedSignalDeriver (calculates moving average)
from streamsim.src.features.simple import SmoothedSignalDeriver
# renderer = ModelLineRenderer (plots raw signal + smoothed model)
import matplotlib.pyplot as plt
from streamsim.src.renderers.model_line import ModelLineRenderer
# detector = ResidualThresholdDetector (simple residual-based change point detection)
from streamsim.src.detectors.residual import ResidualThresholdDetector
# ty together in a simulator
from streamsim.src.core.simulator import StreamingSimulator



def model_demo():
    # Setup
    fig, ax = plt.subplots(figsize=(12, 6))
    setup = PlottingSetup(fig=fig, ax=ax, title="Model line", ylim=(0,15))

    source = create_a_data_source(file_path="a_data.csv", fs=200.0)
    deriver = SmoothedSignalDeriver(window_size = 100)
    detector = ResidualThresholdDetector(threshold=1.5, min_samples=30)


    renderer = ModelLineRenderer(
        show_legend=True,
        signal_label='Raw Signal',
        model_label='Smoothed Model',
        title_template="Latest Value: {feature:.1f}"
    )   


    # 5. Simulator
    sim = StreamingSimulator(
        plotting_setup=setup,
        feature_deriver=deriver,
        change_point_detector=detector,
        renderer=renderer,
        data_source=source,
        window_duration_sec=50,
        max_history=10000,
        interval_ms=1
    )

    logger.info("Starting Model Demo...")
    sim.start()

if __name__ == "__main__":
    model_demo()