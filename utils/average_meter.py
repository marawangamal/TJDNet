class AverageMeter:
    """Computes and stores the average and current value

    Attributes:
        val (float): Last recorded value
        sum (float): Sum of all recorded values
        count (int): Count of recorded values
        avg (float): Running average of recorded values
    """

    def __init__(self, sum: float = 0, count: int = 0, **kwargs):
        """Initialize the AverageMeter"""
        self.reset()
        self.sum = sum
        self.count = count
        self.avg = sum / count if count != 0 else 0

    def reset(self):
        """Reset all statistics"""
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        """Update statistics given new value and optional count

        Args:
            val (float): Value to record
            n (int, optional): Number of values represented by val. Defaults to 1.
        """
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def dump(self):
        """Return the current statistics"""
        return {"sum": self.sum, "count": self.count, "avg": self.avg}
