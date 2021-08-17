import torch

from pynif3d.common.verification import check_pos_int
from pynif3d.log.log_funcs import func_logger


class PositionalEncoding(torch.nn.Module):
    """
    The positional encoding class. It defines several frequency bands and applies
    several frequency responses to the input signal.

    Usage:

    .. code-block:: python

        encoder = PositionalEncoding()
        encoded_points = encoder(points)
    """

    @func_logger
    def __init__(
        self,
        is_include_input: bool = True,
        input_dimensions: int = 3,
        max_frequency: int = 9,
        num_frequency: int = 10,
        frequency_factor: float = 1.0,
        is_log_sampling: bool = True,
        periodic_functions: list = None,
    ) -> torch.nn.Module:
        """
        Args:
            is_include_input (bool): Boolean flags indicating whether to include the
                input signal in the output (True) or not (False). Default is True.
            input_dimensions (int): The dimension of the input signal. Default is 3.
            max_frequency (int): The upper limit of the frequency band. The upper limit
                will be set to `2^max_frequency`. Default is 9.
            num_frequency (int): The number of frequency samples within the spectrum.
                Default is 10.
            frequency_factor (float): The factor to multiply the frequency samples by.
                Default value is 1.0.
            is_log_sampling (bool): Boolean flag indicating whether sampling shall be
                done in log spectrum (True) or not (False). Default is True.
            periodic_functions (list): The periodic functions to be applied per
                frequency. Default is ``[sin, cos]``.
        """
        super().__init__()

        if periodic_functions is None:
            periodic_functions = [torch.sin, torch.cos]

        self.is_include_input = is_include_input
        self.num_frequency = num_frequency
        self.is_log_sampling = is_log_sampling
        self.periodic_functions = periodic_functions
        self.frequency_factor = frequency_factor

        check_pos_int(input_dimensions, "input_dimensions")
        self.input_dimensions = input_dimensions
        check_pos_int(max_frequency, "max_frequency")
        self.max_frequency = max_frequency

        self.freq_bands = torch.linspace(2.0 ** 0, 2.0 ** max_frequency, num_frequency)
        if is_log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, max_frequency, num_frequency)

        self.freq_bands = self.freq_bands * frequency_factor

    def get_dimensions(self):
        output_dimension = 0
        if self.is_include_input:
            output_dimension += self.input_dimensions

        output_dimension += (
            len(self.freq_bands) * len(self.periodic_functions) * self.input_dimensions
        )
        return output_dimension

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Its shape is ``(number_of_samples,
                sample_dimension)``.

        Returns:
            torch.Tensor: Tensor with shape ``(number_of_samples, output_dimensions)``.
            `output_dimensions` can be obtained by calling the `get_dimensions` method.
        """
        res = []

        # Include input if requested
        if self.is_include_input:
            res.append(x)

        for freq in self.freq_bands:
            for per_fn in self.periodic_functions:
                res.append(per_fn(x * freq))

        # Concatenate to create output tensor
        res = torch.cat(res, dim=-1)
        return res
