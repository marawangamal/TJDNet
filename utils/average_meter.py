class AverageMeter:
    """Computes and stores the average and current value

    Attributes:
        val (float): Last recorded value
        sum (float): Sum of all recorded values
        count (int): Count of recorded values
        avg (float): Running average of recorded values
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        """Update statistics given new value and optional count

        Args:
            val (float): Value to record
            n (int, optional): Number of values represented by val. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
