
class CustomData():
    """Generate dummy data."""

    def __init__(self, data):
        """
        Initialize the dataframe.

        Args:
            data (DataFrame): custom data to stream.
        """

        self._data = data
        self._cursor = 0

    def next(self, num_rows=10):
        """
        Get the next chunk of data.

        Args:
            num_rows (int): Number of rows to fetch
        """

        start = self._cursor
        stop = start + num_rows
        self._cursor += num_rows
        return self._data.iloc[start:stop]

    def reset(self):
        """
        Reset the cursor.
        """

        self._cursor = 0

