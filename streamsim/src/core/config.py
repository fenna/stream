"""Configuration of plotting setup for the streaming framework."""

__author__ = "F.Feenstra"


from typing import Any, Dict, Tuple, Optional
import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml

# Load configuration from YAML file
with open("./config.yaml", "r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

# Set up log directory based on config, with fallback to default
try:
    if len(config["log_dir"]) > 0:
        LOG_DIR = Path(config["log_dir"])
    else:
        LOG_DIR = Path.home() / ".streamsim" / "logs"
except KeyError:
    LOG_DIR = Path.home() / ".streamsim" / "logs"

print(f"Log directory set to: ./{LOG_DIR}")
LOG_DIR.mkdir(parents=True, exist_ok=True)



class LoggingSetup:
    """
    Encapsulates logging configuration for the streaming application.
    
    This class provides a structured way to configure logging behavior across
    the entire application. It allows setting log levels, output formats, and
    handlers (file and console) in a centralized manner.
    
    Attributes:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file (str): Path to the log file. If None, no file handler is created.
        console (bool): If True, also output logs to the console (stdout).
        log_format (str): Format string for log messages.
        datefmt (str): Format string for timestamps.
        timestamp_filename (bool): If True, appends a timestamp to the log filename.
    
    Methods:
        setup_logging(): Configures the root logger based on the provided settings.
    """
    
    def __init__(
        self,
        level: int = logging.INFO,
        log_file: str = str(LOG_DIR / f"streamsim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        console: bool = False,
        log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt: str = '%Y-%m-%d %H:%M:%S',
        timestamp_filename: bool = False
    ):
        self.level = level
        self.log_file = log_file
        self.console = console
        self.log_format = log_format
        self.datefmt = datefmt
        self.timestamp_filename = timestamp_filename


    def setup_logging(self) -> None:
        """
        Configure the root logger based on the instance attributes.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        
        if root_logger.handlers:
            root_logger.handlers.clear()

        formatter = logging.Formatter(self.log_format, datefmt=self.datefmt)

        if self.log_file:
            log_path_obj = Path(self.log_file)
            target_dir = log_path_obj.parent
            
            final_log_path = self.log_file
            
            if self.timestamp_filename:
                name = log_path_obj.stem
                ext = log_path_obj.suffix
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_filename = f"{name}_{timestamp}{ext}"
                final_log_path = target_dir / new_filename
            
            try:
                Path(final_log_path).parent.mkdir(parents=True, exist_ok=True)   
                file_handler = logging.FileHandler(str(final_log_path), mode='a')
                file_handler.setLevel(self.level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not create log file '{final_log_path}': {e}", file=sys.stderr)

        if self.console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        noisy_libraries = [
            'matplotlib', 'matplotlib.font_manager', 'PIL', 'PIL.PngImagePlugin', 
            'PIL.Image', 'numpy'
        ]
        
        for lib_name in noisy_libraries:
            logging.getLogger(lib_name).setLevel(logging.WARNING)

        temp_logger = logging.getLogger(__name__)
        temp_logger.info(f"Logging initialized: Level={logging.getLevelName(self.level)}, "
                        f"File={final_log_path if self.log_file else 'None'}, "
                        f"Console={self.console}")
    
    

class PlottingSetup:
    """
    Encapsulates matplotlib figure and axes configuration for streaming visualizations.
    
    This class serves as a centralized container for managing plot setup state,
    including figure/axes references, artist collections, and axis limits. It provides
    a clean interface for initializing and configuring matplotlib plots used in
    real-time streaming applications.
    
    Attributes:
        fig (Any): Matplotlib Figure instance. Typically created via plt.figure()
                   or plt.subplots() before passing to this class.
        ax (Any): Matplotlib Axes instance where data will be rendered. This is
                  the primary plotting surface for streaming visualizations.
        artists (Dict[str, Any]): Dictionary mapping names to matplotlib artists
                                  (lines, markers, patches, etc.) managed by this
                                  setup. Enables organized retrieval and updates
                                  during animation loops. Defaults to empty dict.
        xlim (Tuple[float, float] | None): Optional x-axis bounds as (min, max).
                                           If None, axes auto-scale. Defaults to None.
        ylim (Tuple[float, float] | None): Optional y-axis bounds as (min, max).
                                           If None, axes auto-scale. Defaults to None.
        title (str): Plot title displayed above the axes. 
                     Defaults to "Streaming Data Visualization".
        xlabel (str): Label for the x-axis. Defaults to "Time (s)".
        ylabel (str): Label for the y-axis. Defaults to "Value (units unknown)".

    Methods:
        configure(): Applies the configured title and axis limits to the axes
                     instance. Should be called once during initialization.
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> setup = PlottingSetup(
        ...     fig=fig,
        ...     ax=ax,
        ...     xlim=(0, 100),
        ...     ylim=(-1, 1),
        ...     title="Real-time ECG Monitor"
        ... )
        >>> setup.configure()
        >>> # Later, store artists for tracking
        >>> setup.artists['signal'] = ax.plot([], [])[0]
    
    Note:
        The `Any` type hints for fig and ax are used for flexibility across
        matplotlib backends. In practice, these should be matplotlib.figure.Figure
        and matplotlib.axes.Axes instances respectively.
    
    See Also:
        StreamingRenderer: Base class that consumes PlottingSetup for rendering.
    """
    
    def __init__(
        self,
        fig: Any,
        ax: Any,
        artists: Dict[str, Any] = None,
        xlim: Tuple[float, float] = None,
        ylim: Tuple[float, float] = None,
        xlabel: str = "Time (s)",
        ylabel: str = "Value (units unknown)",
        title: str = "Streaming Data Visualization"
    ):
        """
        Initialize a PlottingSetup instance.
        
        Args:
            fig: Matplotlib Figure instance.
            ax: Matplotlib Axes instance.
            artists: Dictionary of named artists. Defaults to empty dict.
            xlim: Optional x-axis bounds (min, max).
            ylim: Optional y-axis bounds (min, max).
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            title: Plot title text.
        """
        self.fig = fig
        self.ax = ax
        # Handle mutable default safely
        self.artists = artists if artists is not None else {}
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def configure(self) -> None:
        """
        Apply initial configuration to the axes instance.
        
        Sets the plot title and applies fixed axis limits if specified.
        This method should be called once after instantiation to ensure
        consistent initial plot appearance.
        
        Side Effects:
            Modifies the ax instance in-place by setting title and limits.
        
        Example:
            >>> setup = PlottingSetup(fig=fig, ax=ax, title="My Plot")
            >>> setup.configure()  # Title now visible on axes
        """
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)
    
    def __repr__(self) -> str:
        """
        Return a string representation for debugging.
        
        Returns:
            str: Human-readable representation of the instance state.
        """
        return (
            f"PlottingSetup("
            f"fig={self.fig!r}, "
            f"ax={self.ax!r}, "
            f"artists={self.artists!r}, "
            f"xlim={self.xlim!r}, "
            f"ylim={self.ylim!r}, "
            f"xlabel={self.xlabel!r}, "
            f"ylabel={self.ylabel!r}, "
            f"title={self.title!r}"
            f")"
        )
    
