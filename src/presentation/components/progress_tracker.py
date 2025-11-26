"""
Progress tracker component.
"""

import streamlit as st


class ProgressTracker:
    """
    Progress tracker for batch processing.

    Manages progress bar and status text in Streamlit.
    """

    def __init__(self):
        """Initialize progress tracker"""
        self.progress_bar = st.progress(0.0)
        self.status_text = st.empty()
        self.current = 0
        self.total = 0

    def start(self, total: int):
        """
        Start tracking progress.

        Args:
            total: Total number of items to process
        """
        self.total = total
        self.current = 0
        self.update(0, "Starting...")

    def update(self, current: int, message: str = ""):
        """
        Update progress.

        Args:
            current: Current item number
            message: Status message
        """
        self.current = current

        # Update progress bar
        if self.total > 0:
            progress = current / self.total
            self.progress_bar.progress(progress)
        else:
            self.progress_bar.progress(0.0)

        # Update status text
        if message:
            self.status_text.text(f"[{current}/{self.total}] {message}")
        else:
            self.status_text.text(f"[{current}/{self.total}]")

    def complete(self, message: str = "Complete!"):
        """
        Mark progress as complete.

        Args:
            message: Completion message
        """
        self.progress_bar.progress(1.0)
        self.status_text.success(message)

    def error(self, message: str):
        """
        Show error.

        Args:
            message: Error message
        """
        self.status_text.error(message)

    def clear(self):
        """Clear progress display"""
        self.progress_bar.empty()
        self.status_text.empty()
